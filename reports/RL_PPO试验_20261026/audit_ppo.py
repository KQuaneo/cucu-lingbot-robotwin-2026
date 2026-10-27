from pathlib import Path
import json,re,hashlib,subprocess,datetime
import numpy as np
ROOT=Path('/workspace/runtime/experiments/residual_rl_ppo_20261026');PRIOR=Path('/workspace/runtime/experiments/residual_rl_multitask_20261026')
def main():
 s=json.loads((ROOT/'status.json').read_text());assert s['status']=='COMPLETED' and int((ROOT/'exit_code').read_text())==0
 for r in json.loads((ROOT/'training_provenance.json').read_text()):assert hashlib.sha256(Path(r['source']).read_bytes()).hexdigest()==r['sha256']
 tasks=[];episodes=chunks=steps=0;videos=set();maxdelta=0.
 for result in s['task_results']:
  task=result['task'];u=json.loads((ROOT/(task+'_ppo_update.json')).read_text());rows=json.loads((ROOT/(task+'_reused_train_outcomes.json')).read_text());control=json.loads((ROOT/(task+'_control_reference.json')).read_text());after=json.loads((ROOT/(task+'_ppo_after_outcomes.json')).read_text())
  assert len(rows)==len(control)==len(after)==8 and u['nonzero_update'] and u['first_epoch_gradient_matches_analytic']
  assert len({r['seed'] for r in control})==8 and {r['seed'] for r in rows}.isdisjoint({r['seed'] for r in control})
  assert u['epochs']==4 and u['clip']==.2 and u['latent_KL_old_new']<=.020000001
  assert all(np.isfinite(e['gradient_norm']) and np.isfinite(e['loss_before_step']) for e in u['epoch_metrics'])
  old_logprob=-.5*(np.array([r['latent'] for r in rows])**2+np.log(2*np.pi)).sum(-1);assert np.allclose(old_logprob,u['old_logprob'],atol=1e-12)
  stateseq=imageseq=improved=regressed=0
  for c,r in zip(sorted(control,key=lambda x:x['worker']),sorted(after,key=lambda x:x['worker'])):
   assert c['seed']==r['seed'] and c['instruction']==r['instruction'] and c['epsilon']==r['epsilon'] and r['mean']==u['mean_after'];assert r['terminal'] and r['base_frozen']
   old=PRIOR/'trajectories'/c['phase']/f"worker{c['worker']}_seed{c['seed']}"/'chunk_00000.npz';d=ROOT/'trajectories'/r['phase']/f"worker{r['worker']}_seed{r['seed']}"
   with np.load(old) as nc,np.load(d/'chunk_00000.npz') as na:
    stateseq+=int(np.array_equal(nc['observation.state'],na['observation.state']));imageseq+=int(all(np.array_equal(nc[k],na[k]) for k in nc.files if k.startswith('observation.images.')))
   executed=[json.loads(l) for l in (d/'executed_chunks.jsonl').read_text().splitlines()];assert len(executed)==r['chunks']
   for i,p in enumerate(sorted(d.glob('chunk_*.npz'))):
    with np.load(p,allow_pickle=False) as n:
     a=n['applied_action'];b=n['base_action'];assert a.shape==b.shape==(25,14) and np.array_equal(a[:,[6,13]],b[:,[6,13]]);delta=float(np.abs(a-b).max());assert delta<.004001;maxdelta=max(maxdelta,delta)
    assert 0<executed[i]['executed_steps']<=25;chunks+=1;steps+=executed[i]['executed_steps']
   text=re.sub(r'\x1b\[[0-9;]*m','',(ROOT/f"eval_{r['phase']}_gpu{r['worker']}.log").read_text(errors='replace'));m=re.findall(r'Final success rate:\s*(\d+)/(\d+)',text);assert m and int(m[-1][0])==r['reward'] and int(m[-1][1])==1
   paths=re.findall(r'Data has been saved to\s+([^\r\n]+)',text);assert paths;p=Path(paths[-1].strip());p=p if p.is_absolute() else Path('/RoboTwin')/p;videos.update(p.parent.glob('*.mp4'))
   episodes+=1;improved+=int(not c['success'] and r['success']);regressed+=int(c['success'] and not r['success'])
  tasks.append({'task':task,'matched_pairs':8,'initial_state_exact_matches':stateseq,'initial_images_exact_matches':imageseq,'improved_pairs':improved,'regressed_pairs':regressed,'old_logprob_verified':True,'clip_fractions':[e['clip_fraction_after'] for e in u['epoch_metrics']]})
 assert len(tasks)==3 and episodes==len(videos)==24
 assert hashlib.sha256(Path('/RoboTwin/scripts/eval_policy_xpolicylab.py').read_bytes()).hexdigest()=='8f1722bfb25e60ebae17c5bd43a75733e239b63f95e3bd47a7cd434cee224b97'
 ps=subprocess.run(['ps','-eo','pid,args'],capture_output=True,text=True).stdout;alive=[l for l in ps.splitlines() if any('/residual_rl_ppo_20261026/'+x in l for x in ['residual_server.py','run_trial.py','eval_rl.py'])];assert not alive,alive
 out={'verified_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'task_audits':tasks,'new_validation_episodes':episodes,'reused_training_episodes':24,'new_training_episodes':0,'chunks':chunks,'actual_executed_steps':steps,'videos':len(videos),'max_joint_residual_radians':maxdelta,'all_rewards_match_environment_logs':True,'base_frozen':True,'original_eval_script_unchanged':True,'trial_processes_running':[],'exit_code':0};(ROOT/'final_audit.json').write_text(json.dumps(out,indent=2));print(json.dumps(out))
if __name__=='__main__':main()
