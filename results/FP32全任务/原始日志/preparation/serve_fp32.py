import collections, runpy, torch, sys
sys.path.insert(0, '/RoboTwin/experiments/lingbot_vla_v2_6b_robotwin/source/lingbot-vla-v2')
original_cuda = torch.nn.Module.cuda
def audited_cuda(self, *args, **kwargs):
    result = original_cuda(self, *args, **kwargs)
    if type(self).__name__ == 'LingBotVlaV2InferencePolicy':
        parameters = collections.Counter(str(p.dtype) for p in self.parameters() if p.is_floating_point())
        buffers = collections.Counter(str(b.dtype) for b in self.buffers() if b.is_floating_point())
        print('FP32_DTYPE_AUDIT parameters=' + str(dict(parameters)) + ' buffers=' + str(dict(buffers)), flush=True)
        assert parameters and set(parameters) == {'torch.float32'}, parameters
        assert not (set(buffers) - {'torch.float32'}), buffers
    return result
torch.nn.Module.cuda = audited_cuda
runpy.run_module('deploy.lingbot_vla_v2_policy', run_name='__main__')
