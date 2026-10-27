from pathlib import Path
import json,sys,re,subprocess,datetime,hashlib
import numpy as np,torch
ROOT=Path('/workspace/runtime/experiments/conditional_rl_online_20261026');sys.path.insert(0,str(ROOT))
from conditional_model import ResidualNet,features,JOINTS,STD,SCALE

def main():
 status=json.loads((ROOT/'status.json').read_text());assert status['status']=='COMPLETED' and int((ROOT/'exit_code').read_text())==0
 states={v:torch.load(ROOT/f'actor_v{v}.pt',weights_only=True,map_location='cpu') for v in range(4)};actors={};changes=[];opt_steps=[]
 for v,sd in states.items():m=ResidualNet();m.load_state_dict(sd);m.eval();actors[v]=m
 assert sum(p.numel() for p in actors[0].parameters())==5900
 for v,row in enumerate(status['round_results']):
  change=float(torch.sqrt(sum((states[v+1][k]-x).square().sum() for k,x in states[v].items())));assert np.isclose(change,row['parameter_change_norm'],atol=1e-12);changes.append(change)
  opt=torch.load(ROOT/f'optimizer_v{v+1}.pt',weights_only=True,map_location='cpu');steps=[float(x['step']) for x in opt['state'].values()];opt_steps.append(max(steps) if steps else 0)
 assert opt_steps==sorted(opt_steps) and opt_steps[-1]>=4
 episodes=chunks=steps=0;maxdelta=0.;videos=set();phase_seeds={};observed_means=[]
 for p in ROOT.glob('trajectories/*/*/episode.json'):
  row=json.loads(p.read_text());assert row['terminal'] and row['base_frozen'] and row['conditional'];d=p.parent;version=row['policy_version'];phase_seeds.setdefault(row['phase'],set()).add(row['seed']);executed=[json.loads(l) for l in (d/'executed_chunks.jsonl').read_text().splitlines()];assert len(executed)==row['chunks']
  for idx,q in enumerate(sorted(d.glob('chunk_*.npz'))):
   with np.load(q,allow_pickle=False) as n:
    x=n['features'];z=n['latent'];mu=n['old_mean'];assert x.shape==(171,) and np.allclose(features(n,n['base_action'],idx),x,atol=1e-12)
    with torch.no_grad():pred=actors[version](torch.tensor(x)).numpy()
    assert np.allclose(pred,mu,atol=1e-12);assert np.allclose(z,mu+STD*n['epsilon'],atol=1e-12)
    a=n['applied_action'];b=n['base_action'];assert a.shape==b.shape==(25,14) and np.array_equal(a[:,[6,13]],b[:,[6,13]])
    delta=(SCALE*np.tanh(z)).astype(np.float32);expected=b.copy();expected[:,JOINTS]+=delta;assert np.array_equal(a,expected)
    maxdelta=max(maxdelta,float(np.abs(a-b).max()));assert maxdelta<.020001
    if version==3:observed_means.append(mu)
   assert 0<executed[idx]['executed_steps']<=25;steps+=executed[idx]['executed_steps'];chunks+=1
  text=re.sub(r'\x1b\[[0-9;]*m','',(ROOT/f"eval_{row['phase']}_gpu{row['worker']}.log").read_text(errors='replace'));m=re.findall(r'Final success rate:\s*(\d+)/(\d+)',text);assert m and int(m[-1][0])==row['reward'] and int(m[-1][1])==1
  paths=re.findall(r'Data has been saved to\s+([^\r\n]+)',text);assert paths;v=Path(paths[-1].strip());v=v if v.is_absolute() else Path('/RoboTwin')/v;videos.update(v.parent.glob('*.mp4'));episodes+=1
 for i in range(3):
  for j in range(i+1,3):assert phase_seeds['online_round'+str(i)].isdisjoint(phase_seeds['online_round'+str(j)])
  assert phase_seeds['online_round'+str(i)].isdisjoint(phase_seeds['validation_after'])
 conditional_span=float(np.ptp(np.array(observed_means),axis=0).max());assert conditional_span>1e-9
 paired_info=None
 supplement=ROOT/'supplement_status.json'
 if supplement.exists() and json.loads(supplement.read_text())['status']=='COMPLETED':
  assert int((ROOT/'supplement_exit_code').read_text())==0;paired=json.loads((ROOT/'conditional_paired_validation.json').read_text());stateeq=imageeq=eps_eq=0
  for row in paired:
   assert row['seed_equal'] and row['instruction_equal'];worker=row['worker'];seed=row['seed'];pc=ROOT/'trajectories/validation_control_v0'/f'worker{worker}_seed{seed}'/'chunk_00000.npz';pa=ROOT/'trajectories/validation_after'/f'worker{worker}_seed{seed}'/'chunk_00000.npz'
   with np.load(pc) as c,np.load(pa) as a:stateeq+=int(np.array_equal(c['observation.state'],a['observation.state']));imageeq+=int(all(np.array_equal(c[k],a[k]) for k in c.files if k.startswith('observation.images.')));eps_eq+=int(np.array_equal(c['epsilon'],a['epsilon']))
  paired_info={'pairs':len(paired),'initial_state_exact_matches':stateeq,'initial_images_exact_matches':imageeq,'first_chunk_epsilon_exact_matches':eps_eq,'control_successes':sum(int(r['control_success']) for r in paired),'trained_successes':sum(int(r['trained_success']) for r in paired),'improved_pairs':sum(not r['control_success'] and r['trained_success'] for r in paired),'regressed_pairs':sum(r['control_success'] and not r['trained_success'] for r in paired)}
 assert episodes==len(videos)==(20 if paired_info else 16)
 assert hashlib.sha256(Path('/RoboTwin/scripts/eval_policy_xpolicylab.py').read_bytes()).hexdigest()=='8f1722bfb25e60ebae17c5bd43a75733e239b63f95e3bd47a7cd434cee224b97'
 ps=subprocess.run(['ps','-eo','pid,args'],capture_output=True,text=True).stdout;alive=[l for l in ps.splitlines() if any('/conditional_rl_online_20261026/'+x in l for x in ['conditional_server.py','run_trial.py','eval_rl.py','supplement_control.py'])];assert not alive,alive
 out={'verified_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'trainable_parameters':5900,'input_features':171,'fresh_online_training_episodes':12,'nonzero_updates':sum(c>0 for c in changes),'parameter_change_norms':changes,'optimizer_step_counters':opt_steps,'all_old_means_recomputed_from_correct_checkpoint':True,'conditional_mean_span_on_actual_observations':conditional_span,'all_rewards_match_environment_logs':True,'episodes':episodes,'chunks':chunks,'actual_executed_steps':steps,'videos':len(videos),'max_joint_residual_radians':maxdelta,'paired_validation':paired_info,'original_eval_unchanged':True,'base_frozen':True,'trial_processes_running':[],'exit_code':0};(ROOT/'final_audit.json').write_text(json.dumps(out,indent=2));print(json.dumps(out))
if __name__=='__main__':main()
