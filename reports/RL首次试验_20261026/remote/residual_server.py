"""Frozen LingBot + episode-level Gaussian latent residual policy; research smoke only."""
import os,sys,json,time,argparse,hashlib
from pathlib import Path
import numpy as np
import torch
SOURCE='/RoboTwin/experiments/lingbot_vla_v2_6b_robotwin/source/lingbot-vla-v2'
sys.path.insert(0,SOURCE)
from deploy.lingbot_vla_v2_policy import LingbotVLAv2Server
from deploy.websocket_policy_server import WebsocketPolicyServer
JOINTS=np.array([0,1,2,3,4,5,7,8,9,10,11,12])
class ResidualPolicy:
 def __init__(self,base,root,worker):
  self.base=base;self.root=Path(root);self.worker=worker;self.record=None
  for p in base.vla.parameters():p.requires_grad_(False)
 def infer(self,obs):
  if obs.get('rl_event')=='executed':
   count=int(obs['count']);assert 0<count<=25
   with (self.episode_dir/'executed_chunks.jsonl').open('a') as f:f.write(json.dumps({'chunk_index':self.record['chunks']-1,'executed_steps':count,'success_after_chunk':bool(obs['success'])})+'\n')
   return {'rl_execution_recorded':True}
  if obs.get('rl_event')=='end':
   assert self.record is not None
   assert int(obs['seed'])==self.record['seed']
   self.record.update(success=bool(obs['success']),reward=float(bool(obs['success'])),terminal=True,ended_utc=time.time())
   self.save();return {'rl_recorded':True}
  if obs.get('reset'):
   phase=json.loads((self.root/'phase.json').read_text());meta=obs['rl_case_meta'];seed=int(meta['seed'])
   mu=np.asarray(phase['mean'],dtype=np.float64);rng=np.random.default_rng(seed+20261026);eps=rng.normal(size=12);z=mu+eps
   self.delta=(0.004*np.tanh(z)).astype(np.float32)
   self.episode_dir=self.root/'trajectories'/phase['phase']/f'worker{self.worker}_seed{seed}';self.episode_dir.mkdir(parents=True,exist_ok=False)
   self.record=dict(meta,phase=phase['phase'],worker=self.worker,mean=mu.tolist(),epsilon=eps.tolist(),latent=z.tolist(),delta_radians=self.delta.tolist(),chunks=0,terminal=False,base_frozen=True,residual_scale=0.004,started_utc=time.time())
   torch.manual_seed(seed+42);torch.cuda.manual_seed_all(seed+42);self.save()
   return self.base.infer(obs)
  assert self.record is not None
  with torch.inference_mode():result=self.base.infer(obs)
  actions=np.asarray(result['action'],dtype=np.float32);assert actions.ndim==2 and actions.shape[1]==14,actions.shape
  changed=actions.copy();changed[:,JOINTS]+=self.delta
  assert np.isfinite(changed).all() and np.array_equal(changed[:,[6,13]],actions[:,[6,13]])
  arrays={k:np.asarray(v) for k,v in obs.items() if isinstance(v,(np.ndarray,list,tuple))}
  arrays.update(base_action=actions,applied_action=changed,delta_radians=self.delta)
  np.savez_compressed(self.episode_dir/f"chunk_{self.record['chunks']:05d}.npz",**arrays)
  self.record['chunks']+=1;self.save();result['action']=changed;return result
 def save(self):
  p=self.episode_dir/'episode.json';tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(self.record,indent=2));tmp.replace(p)
def main():
 p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--worker',type=int,required=True);p.add_argument('--port',type=int,required=True);a=p.parse_args()
 base=LingbotVLAv2Server('/workspace/runtime/models/full_sft_6000',use_length=25,chunk_ret=True,use_bf16=False,use_fp32=True,use_compile=False)
 policy=ResidualPolicy(base,a.root,a.worker)
 assert all(not p.requires_grad for p in base.vla.parameters())
 print('BASE_FROZEN_FP32_READY',flush=True)
 WebsocketPolicyServer(policy,host='127.0.0.1',port=a.port).serve_forever()
if __name__=='__main__':main()
