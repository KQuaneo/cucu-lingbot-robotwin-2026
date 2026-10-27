import torch,numpy as np
from torch import nn
JOINTS=np.array([0,1,2,3,4,5,7,8,9,10,11,12])
STD=.25;SCALE=.02;INPUT_DIM=171
class ResidualNet(nn.Module):
 def __init__(self):
  super().__init__();self.net=nn.Sequential(nn.Linear(INPUT_DIM,32),nn.Tanh(),nn.Linear(32,12));self.double()
 def forward(self,x):return .5*torch.tanh(self.net(x))
def initial_model():
 torch.manual_seed(42);m=ResidualNet();nn.init.zeros_(m.net[-1].weight);nn.init.zeros_(m.net[-1].bias);return m
def features(obs,actions,chunk):
 images=[]
 for k in ['observation.images.cam_high','observation.images.cam_left_wrist','observation.images.cam_right_wrist']:
  a=torch.tensor(np.array(obs[k],copy=True),dtype=torch.float32).permute(2,0,1)[None]/255.
  images.extend((torch.nn.functional.adaptive_avg_pool2d(a,(4,4)).flatten()*2-1).tolist())
 state=np.asarray(obs['observation.state'],dtype=np.float64).copy();state[JOINTS]/=np.pi;state=np.clip(state,-2,2)
 x=np.concatenate([np.array(images),state,np.clip(actions[0,JOINTS]/np.pi,-2,2),[min(chunk/16.,1.)]])
 assert x.shape==(INPUT_DIM,) and np.isfinite(x).all();return x
