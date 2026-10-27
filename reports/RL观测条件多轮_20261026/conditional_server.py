import os,sys,json,time,argparse
from pathlib import Path
import numpy as np,torch
from conditional_model import ResidualNet,features,JOINTS,STD,SCALE
SOURCE='/RoboTwin/experiments/lingbot_vla_v2_6b_robotwin/source/lingbot-vla-v2';sys.path.insert(0,SOURCE)
from deploy.lingbot_vla_v2_policy import LingbotVLAv2Server
from deploy.websocket_policy_server import WebsocketPolicyServer
class ConditionalPolicy:
 def __init__(self,base,root,worker):
  self.base=base;self.root=Path(root);self.worker=worker;self.actor=ResidualNet();self.record=None
  for p in base.vla.parameters():p.requires_grad_(False)
 def save(self):
  p=self.directory/'episode.json';t=p.with_suffix('.tmp');t.write_text(json.dumps(self.record,indent=2));t.replace(p)
 def infer(self,obs):
  if obs.get('rl_event')=='executed':
   count=int(obs['count']);assert 0<count<=25
   with (self.directory/'executed_chunks.jsonl').open('a') as f:f.write(json.dumps({'chunk_index':self.record['chunks']-1,'executed_steps':count,'success_after_chunk':bool(obs['success'])})+'\n')
   return {'rl_execution_recorded':True}
  if obs.get('rl_event')=='end':
   assert int(obs['seed'])==self.record['seed'];self.record.update(terminal=True,success=bool(obs['success']),reward=float(bool(obs['success'])));self.save();return {'rl_recorded':True}
  if obs.get('reset'):
   phase=json.loads((self.root/'phase.json').read_text());version=int(phase['version']);self.actor.load_state_dict(torch.load(self.root/f'actor_v{version}.pt',map_location='cpu',weights_only=True));self.actor.eval()
   meta=obs['rl_case_meta'];seed=int(meta['seed']);self.rng=np.random.default_rng(seed+20261026);self.directory=self.root/'trajectories'/phase['phase']/f'worker{self.worker}_seed{seed}';self.directory.mkdir(parents=True,exist_ok=False)
   self.record=dict(meta,phase=phase['phase'],worker=self.worker,policy_version=version,chunks=0,terminal=False,base_frozen=True,conditional=True,std=STD,scale=SCALE)
   torch.manual_seed(seed+42);torch.cuda.manual_seed_all(seed+42);self.save();return self.base.infer(obs)
  with torch.inference_mode():result=self.base.infer(obs)
  actions=np.asarray(result['action'],dtype=np.float32);assert actions.shape==(25,14)
  x=features(obs,actions,self.record['chunks'])
  with torch.no_grad():mean=self.actor(torch.tensor(x)).numpy()
  eps=self.rng.normal(size=12);z=mean+STD*eps;delta=(SCALE*np.tanh(z)).astype(np.float32);changed=actions.copy();changed[:,JOINTS]+=delta
  assert np.isfinite(changed).all() and np.array_equal(changed[:,[6,13]],actions[:,[6,13]])
  arrays={k:np.asarray(v) for k,v in obs.items() if isinstance(v,(np.ndarray,list,tuple))};arrays.update(features=x,old_mean=mean,epsilon=eps,latent=z,base_action=actions,applied_action=changed,delta_radians=delta)
  np.savez_compressed(self.directory/f"chunk_{self.record['chunks']:05d}.npz",**arrays);self.record['chunks']+=1;self.save();result['action']=changed;return result
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--worker',type=int,required=True);p.add_argument('--port',type=int,required=True);a=p.parse_args()
 base=LingbotVLAv2Server('/workspace/runtime/models/full_sft_6000',use_length=25,chunk_ret=True,use_bf16=False,use_fp32=True,use_compile=False);policy=ConditionalPolicy(base,a.root,a.worker);assert all(not p.requires_grad for p in base.vla.parameters());print('BASE_FROZEN_FP32_READY',flush=True);WebsocketPolicyServer(policy,host='127.0.0.1',port=a.port).serve_forever()
