"""CUDA correctness gate for the single-rank FSDP2 CPU-offload path."""
import copy, json, os, tempfile
import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import CPUOffloadPolicy, MixedPrecisionPolicy, fully_shard
from torch.distributed.tensor import DTensor

dist.init_process_group('nccl')
assert dist.get_world_size() == 1
assert torch.__version__.split('+')[0] == '2.8.0'
torch.set_float32_matmul_precision("highest")
torch.backends.cuda.matmul.allow_tf32=False
torch.manual_seed(12)
model=torch.nn.Sequential(torch.nn.Linear(16,32),torch.nn.GELU(),torch.nn.Linear(32,8))
reference=copy.deepcopy(model).cuda()
mesh=init_device_mesh('cuda',(1,))
policy=dict(mesh=mesh,mp_policy=MixedPrecisionPolicy(param_dtype=torch.float32,reduce_dtype=torch.float32,output_dtype=torch.float32),offload_policy=CPUOffloadPolicy())
fully_shard(model[0],**policy);fully_shard(model[2],**policy);fully_shard(model,**policy)
optim=torch.optim.AdamW(model.parameters(),lr=1e-3,foreach=False)
ref_optim=torch.optim.AdamW(reference.parameters(),lr=1e-3,foreach=False)
losses=[]
for _ in range(2):
    x=torch.randn(3,16,device='cuda'); y=torch.randn(3,8,device='cuda')
    loss=(model(x).float()-y).square().mean()/2
    with torch.autocast('cuda',enabled=False):
        ref_loss=(reference(x).float()-y).square().mean()/2
    loss.backward();ref_loss.backward();losses.append(loss.item())
for p,q in zip(model.parameters(),reference.parameters()):
    grad=p.grad.to_local() if isinstance(p.grad,DTensor) else p.grad
    assert grad.device.type=='cpu' and torch.isfinite(grad).all()
    torch.testing.assert_close(grad,q.grad.cpu(),rtol=1e-5,atol=1e-6)
optim.step();ref_optim.step();optim.zero_grad(set_to_none=True)
for p,q in zip(model.parameters(),reference.parameters()):
    tensor=p.to_local() if isinstance(p,DTensor) else p
    assert tensor.device.type=='cpu'
    torch.testing.assert_close(tensor,q.cpu(),rtol=0.005,atol=0.002)
state_model,state_optim=get_state_dict(model,optim)
state={'model':state_model,'optimizer':state_optim}
folder=tempfile.mkdtemp(prefix='full-sft-offload-dcp-')
dcp.save(state,checkpoint_id=folder)
before=[p.detach().to_local().clone() for p in model.parameters()]
with torch.no_grad():
    for p in model.parameters():p.add_(0.1)
state_model,state_optim=get_state_dict(model,optim)
state={'model':state_model,'optimizer':state_optim}
dcp.load(state,checkpoint_id=folder)
set_state_dict(model,optim,model_state_dict=state['model'],optim_state_dict=state['optimizer'])
for p,b in zip(model.parameters(),before):torch.testing.assert_close(p.to_local(),b,rtol=0,atol=0)
assert optim.state and all('exp_avg' in v and 'exp_avg_sq' in v for v in optim.state.values())
print(json.dumps({'gate':'single_rank_fsdp2_cpu_offload','accumulation':2,'finite_loss':losses,'grad_reference_match':True,'cpu_adamw':True,'dcp_restore':True,'torch':torch.__version__}))
dist.destroy_process_group()
