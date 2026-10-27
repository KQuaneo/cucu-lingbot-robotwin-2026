from pathlib import Path
import torch,numpy as np,copy
from conditional_model import ResidualNet,STD,INPUT_DIM

def update(root,rows,version):
 actor=ResidualNet();actor.load_state_dict(torch.load(root/f'actor_v{version}.pt',weights_only=True,map_location='cpu'));opt=torch.optim.Adam(actor.parameters(),lr=1e-3)
 if (root/f'optimizer_v{version}.pt').exists():opt.load_state_dict(torch.load(root/f'optimizer_v{version}.pt',map_location='cpu',weights_only=True))
 rewards=np.array([r['reward'] for r in rows]);advantages=rewards-(rewards.sum()-rewards)/(len(rows)-1);xs=[];zs=[];means=[];advs=[]
 for row,adv in zip(rows,advantages):
  d=root/'trajectories'/row['phase']/f"worker{row['worker']}_seed{row['seed']}"
  assert row['terminal'] and row['policy_version']==version
  for p in sorted(d.glob('chunk_*.npz')):
   with np.load(p,allow_pickle=False) as n:xs.append(n['features']);zs.append(n['latent']);means.append(n['old_mean']);advs.append(adv)
 x=torch.tensor(np.array(xs));z=torch.tensor(np.array(zs));oldmean=torch.tensor(np.array(means));adv=torch.tensor(advs);oldlog=torch.distributions.Normal(oldmean,STD).log_prob(z).sum(-1).detach();assert torch.allclose(actor(x),oldmean,atol=1e-12)
 before={k:v.clone() for k,v in actor.state_dict().items()};metrics=[]
 for epoch in range(2):
  prior={k:v.clone() for k,v in actor.state_dict().items()};prior_opt=copy.deepcopy(opt.state_dict());opt.zero_grad();mu=actor(x);ratio=(torch.distributions.Normal(mu,STD).log_prob(z).sum(-1)-oldlog).exp();loss=-torch.minimum(ratio*adv,ratio.clamp(.8,1.2)*adv).sum()/len(rows);loss.backward()
  grad=float(torch.nn.utils.clip_grad_norm_(actor.parameters(),1.));assert np.isfinite(grad);opt.step()
  with torch.no_grad():kl=float(((actor(x)-oldmean).square().sum(-1)/(2*STD**2)).mean())
  rolled_back=kl>.02
  if rolled_back:actor.load_state_dict(prior);opt.load_state_dict(prior_opt)
  metrics.append({'epoch':epoch+1,'loss':float(loss.detach()),'gradient_norm':grad,'proposed_KL':kl,'rolled_back':rolled_back})
  if rolled_back:break
 change=float(torch.sqrt(sum((actor.state_dict()[k]-v).square().sum() for k,v in before.items())))
 probe=x[:1].clone();probe[:,144]+=.1
 with torch.no_grad():conditional_difference=float((actor(x[:1])-actor(probe)).abs().max())
 output=root/f'actor_v{version+1}.pt';tmp=output.with_suffix('.tmp');torch.save(actor.state_dict(),tmp);tmp.replace(output);torch.save(opt.state_dict(),root/f'optimizer_v{version+1}.pt')
 return {'version_before':version,'version_after':version+1,'rewards':rewards.tolist(),'advantages':advantages.tolist(),'episodes':len(rows),'chunks':len(xs),'parameter_change_norm':change,'nonzero_update':change>0,'conditional_feature_probe_difference':conditional_difference,'metrics':metrics,'trainable_parameters':sum(p.numel() for p in actor.parameters()),'optimizer_reset_each_round':False,'std':STD,'algorithm':'per-chunk clipped PPO actor with episode Monte Carlo LOO advantages; no critic'}
