from pathlib import Path
import json,re,hashlib,subprocess,datetime
import numpy as np
ROOT=Path('/workspace/runtime/experiments/residual_rl_multitask_20261026')
def main():
 status=json.loads((ROOT/'status.json').read_text());assert status['status']=='COMPLETED';assert int((ROOT/'exit_code').read_text())==0
 results=json.loads((ROOT/'task_results.json').read_text());audits=[]
 for result in results:
  task=result['task'];u=json.loads((ROOT/(task+'_policy_update.json')).read_text());rows=json.loads((ROOT/(task+'_train0_outcomes.json')).read_text())
  extra=ROOT/(task+'_train1_outcomes.json')
  if extra.exists():rows+=json.loads(extra.read_text())
  rewards=np.array([r['reward'] for r in rows]);z=np.array([r['latent'] for r in rows]);means=np.array([r['mean'] for r in rows]);assert np.array_equal(means,np.zeros_like(means))
  adv=rewards-(rewards.sum()-rewards)/(len(rows)-1);gradient=-(adv[:,None]*z).mean(0);assert np.allclose(gradient,u['gradient'],atol=1e-12)
  assert all(r['task_name']==task and r['terminal'] for r in rows)
  a={'task':task,'analytic_gradient_matches':True,'training_rewards':rewards.tolist(),'update_nonzero':u['nonzero_update'],'status':result['status']}
  if u['nonzero_update']:
   control=json.loads((ROOT/(task+'_control_outcomes.json')).read_text());after=json.loads((ROOT/(task+'_after_outcomes.json')).read_text());pairs=json.loads((ROOT/(task+'_paired_validation.json')).read_text())
   assert len(control)==len(after)==len(pairs)==8
   assert len({r['seed'] for r in control})==8
   assert {r['seed'] for r in rows}.isdisjoint({r['seed'] for r in control})
   first_state_equal=[];first_images_equal=[]
   for c,r,pair in zip(sorted(control,key=lambda x:x['worker']),sorted(after,key=lambda x:x['worker']),sorted(pairs,key=lambda x:x['worker'])):
    assert c['seed']==r['seed'] and c['instruction']==r['instruction'] and c['epsilon']==r['epsilon']
    assert np.array_equal(c['mean'],np.zeros(12)) and np.array_equal(r['mean'],u['mean_after'])
    pc=ROOT/'trajectories'/c['phase']/f"worker{c['worker']}_seed{c['seed']}"/'chunk_00000.npz';pa=ROOT/'trajectories'/r['phase']/f"worker{r['worker']}_seed{r['seed']}"/'chunk_00000.npz'
    with np.load(pc,allow_pickle=False) as nc,np.load(pa,allow_pickle=False) as na:
     first_state_equal.append(np.array_equal(nc['observation.state'],na['observation.state']))
     imagekeys=[k for k in nc.files if k.startswith('observation.images.')];first_images_equal.append(all(np.array_equal(nc[k],na[k]) for k in imagekeys))
   a.update(matched_pairs=8,initial_state_exact_matches=sum(first_state_equal),initial_images_exact_matches=sum(first_images_equal),improved_pairs=sum(not p['control_success'] and p['after_success'] for p in pairs),regressed_pairs=sum(p['control_success'] and not p['after_success'] for p in pairs),heldout_seeds_disjoint=True)
  audits.append(a)
 episodes=chunks=steps=0;videos=set();max_delta=0.
 for p in ROOT.glob('trajectories/*/*/episode.json'):
  r=json.loads(p.read_text());assert r['terminal'] and r['base_frozen'];executed=[json.loads(l) for l in (p.parent/'executed_chunks.jsonl').read_text().splitlines()];assert len(executed)==r['chunks']
  for i,q in enumerate(sorted(p.parent.glob('chunk_*.npz'))):
   with np.load(q,allow_pickle=False) as n:
    aa=n['applied_action'];bb=n['base_action'];assert aa.shape==bb.shape==(25,14);assert np.array_equal(aa[:,[6,13]],bb[:,[6,13]]);d=float(np.abs(aa-bb).max());assert d<.004001;max_delta=max(max_delta,d)
   assert 0<executed[i]['executed_steps']<=25;chunks+=1;steps+=executed[i]['executed_steps']
  text=re.sub(r'\x1b\[[0-9;]*m','',(ROOT/f"eval_{r['phase']}_gpu{r['worker']}.log").read_text(errors='replace'));m=re.findall(r'Final success rate:\s*(\d+)/(\d+)',text);assert m and int(m[-1][0])==r['reward'] and int(m[-1][1])==1
  paths=re.findall(r'Data has been saved to\s+([^\r\n]+)',text);assert paths;v=Path(paths[-1].strip());v=v if v.is_absolute() else Path('/RoboTwin')/v;videos.update(v.parent.glob('*.mp4'));episodes+=1
 assert len(videos)==episodes
 assert hashlib.sha256(Path('/RoboTwin/scripts/eval_policy_xpolicylab.py').read_bytes()).hexdigest()=='8f1722bfb25e60ebae17c5bd43a75733e239b63f95e3bd47a7cd434cee224b97'
 ps=subprocess.run(['ps','-eo','pid,args'],capture_output=True,text=True).stdout;alive=[l for l in ps.splitlines() if any('/residual_rl_multitask_20261026/'+x in l for x in ['residual_server.py','run_trial.py','eval_rl.py'])];assert not alive,alive
 out={'verified_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'task_audits':audits,'episodes':episodes,'chunks':chunks,'actual_executed_steps':steps,'videos':len(videos),'max_abs_joint_residual_radians':max_delta,'all_environment_rewards_match_logs':True,'base_frozen':True,'original_eval_script_unchanged':True,'exit_code':0,'trial_processes_running':[]};(ROOT/'final_audit.json').write_text(json.dumps(out,indent=2));print(json.dumps(out))
if __name__=='__main__':main()
