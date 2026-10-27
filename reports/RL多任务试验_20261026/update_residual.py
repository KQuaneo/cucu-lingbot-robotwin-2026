"""One REINFORCE update of Gaussian episode-level latent mean, fixed unit variance."""
import json,math
from pathlib import Path
import torch
def update(rows,learning_rate=0.2,max_norm=0.2):
 assert len(rows)>=2 and all(r['terminal'] and r['chunks']>0 for r in rows)
 means=torch.tensor([r['mean'] for r in rows],dtype=torch.float64);assert torch.equal(means,means[0].expand_as(means))
 mu=torch.nn.Parameter(means[0].clone());z=torch.tensor([r['latent'] for r in rows],dtype=torch.float64)
 rewards=torch.tensor([r['reward'] for r in rows],dtype=torch.float64)
 assert set(rewards.tolist())<={0.,1.}
 # Leave-one-out baseline: each baseline excludes this rollout's own reward.
 advantage=rewards-(rewards.sum()-rewards)/(len(rows)-1)
 logprob=torch.distributions.Normal(mu,torch.ones_like(mu)).log_prob(z).sum(-1)
 loss=-(advantage.detach()*logprob).mean();loss.backward();grad=mu.grad.detach().clone();assert torch.isfinite(grad).all()
 proposed=-learning_rate*grad;norm=float(proposed.norm());step=proposed*min(1.,max_norm/max(norm,1e-12))
 with torch.no_grad():mu.add_(step)
 return dict(algorithm='episode-latent Gaussian REINFORCE with leave-one-out baseline',learning_rate=learning_rate,rewards=rewards.tolist(),advantages=advantage.tolist(),loss=float(loss.detach()),gradient=grad.tolist(),gradient_norm=float(grad.norm()),mean_before=means[0].tolist(),mean_after=mu.detach().tolist(),parameter_change_norm=float(step.norm()),latent_KL_old_new=float(.5*step.square().sum()),nonzero_update=bool(step.norm()>0),base_model_updated=False)
def self_test():
 rows=[dict(terminal=True,chunks=1,mean=[0.]*12,latent=[v]*12,reward=r) for v,r in [(1.,1.),(-1.,0.)]]
 r=update(rows);assert r['nonzero_update'] and all(v>0 for v in r['mean_after'])
 same=update([dict(x,reward=1.) for x in rows]);assert not same['nonzero_update']
 return {'passed':True,'positive_reward_moves_mean_towards_success':True,'equal_rewards_do_not_fake_update':True}
if __name__=='__main__':print(json.dumps(self_test()))
