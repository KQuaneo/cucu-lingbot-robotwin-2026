"""Clipped PPO actor objective on an episode-level Gaussian latent; no critic."""
import torch,json

def update(rows,epochs=4,learning_rate=.03,clip=.2,max_mean_norm=.2):
 assert len(rows)>=2 and all(r['terminal'] and r['chunks']>0 for r in rows)
 old=torch.tensor([r['mean'] for r in rows],dtype=torch.float64);assert torch.equal(old,old[0].expand_as(old)) and bool((old==0).all())
 z=torch.tensor([r['latent'] for r in rows],dtype=torch.float64);reward=torch.tensor([r['reward'] for r in rows],dtype=torch.float64);assert set(reward.tolist())<={0.,1.}
 advantage=reward-(reward.sum()-reward)/(len(rows)-1)
 mu=torch.nn.Parameter(old[0].clone());old_logprob=torch.distributions.Normal(old[0],torch.ones(12,dtype=torch.float64)).log_prob(z).sum(-1).detach()
 opt=torch.optim.Adam([mu],lr=learning_rate);history=[]
 for epoch in range(epochs):
  opt.zero_grad();logprob=torch.distributions.Normal(mu,torch.ones_like(mu)).log_prob(z).sum(-1);ratio=(logprob-old_logprob).exp()
  loss=-torch.minimum(ratio*advantage,ratio.clamp(1-clip,1+clip)*advantage).mean();loss.backward();grad=mu.grad.detach().clone();assert torch.isfinite(grad).all()
  if epoch==0:
   analytic=-(advantage[:,None]*(z-old[0])).mean(0);assert torch.allclose(grad,analytic,atol=1e-12)
  norm=float(torch.nn.utils.clip_grad_norm_([mu],1.));opt.step()
  with torch.no_grad():
   shift=mu-old[0];mu.copy_(old[0]+shift*min(1.,max_mean_norm/max(float(shift.norm()),1e-12)))
   new_ratio=(torch.distributions.Normal(mu,torch.ones_like(mu)).log_prob(z).sum(-1)-old_logprob).exp()
   history.append({'epoch':epoch+1,'loss_before_step':float(loss.detach()),'gradient_norm':norm,'ratio_min_after':float(new_ratio.min()),'ratio_max_after':float(new_ratio.max()),'clip_fraction_after':float(((new_ratio<1-clip)|(new_ratio>1+clip)).double().mean()),'latent_KL_old_new':float(.5*(mu-old[0]).square().sum()),'mean_after':mu.tolist()})
 return {'algorithm':'clipped PPO actor objective on episode-level Gaussian residual latent','advantage_estimator':'Monte Carlo terminal reward with leave-one-out baseline; no value critic','rewards':reward.tolist(),'advantages':advantage.tolist(),'old_logprob':old_logprob.tolist(),'epochs':epochs,'learning_rate':learning_rate,'clip':clip,'mean_before':old[0].tolist(),'mean_after':mu.detach().tolist(),'parameter_change_norm':float((mu.detach()-old[0]).norm()),'latent_KL_old_new':float(.5*(mu.detach()-old[0]).square().sum()),'nonzero_update':bool((mu.detach()-old[0]).norm()>0),'epoch_metrics':history,'base_model_updated':False,'first_epoch_gradient_matches_analytic':True}

def self_test():
 rows=[dict(terminal=True,chunks=1,mean=[0.]*12,latent=[z]*12,reward=r) for z,r in [(1.,1.),(-1.,0.)]]
 r=update(rows);assert r['nonzero_update'] and all(v>0 for v in r['mean_after']);assert r['latent_KL_old_new']<=.020000001
 equal=update([dict(x,reward=1.) for x in rows]);assert not equal['nonzero_update']
 ratio=torch.tensor([2.,.1],dtype=torch.float64);adv=torch.tensor([1.,-1.],dtype=torch.float64);value=torch.minimum(ratio*adv,ratio.clamp(.8,1.2)*adv);assert torch.allclose(value,torch.tensor([1.2,-.8],dtype=torch.float64))
 return {'passed':True,'clipping_both_advantage_signs_correct':True,'first_epoch_gradient_analytic':True,'equal_rewards_zero_update':True,'KL_bound':True}
if __name__=='__main__':print(json.dumps(self_test()))
