import torch
from torch import Tensor
from torch._ops import ops
from torch.nn import functional as F

import IPython


class ExtendedQuantizedOpsStub(torch.nn.Module):
    def __init__(self):
        super(ExtendedQuantizedOpsStub, self).__init__()

    def forward(self, x):
        raise RuntimeError("FloatFunctional is not intended to use the " +
                           "'forward'. Please use the underlying operation")

    def softmax(self, x: Tensor, dim=-1) -> Tensor:
        return F.softmax(x, dim=dim)

    def matmul(self, x: Tensor, y: Tensor) -> Tensor:
        return torch.matmul(x, y)

        
class ExtendedQuantizedOpsObserver(torch.nn.Module):
    def __init__(self):
        super(ExtendedQuantizedOpsObserver, self).__init__()

    def forward(self, x):
        raise RuntimeError("FloatFunctional is not intended to use the " +
                           "'forward'. Please use the underlying operation")

    def softmax(self, x: Tensor, dim=-1) -> Tensor:
        r = F.softmax(x, dim=dim)
        r = self.activation_post_process(r)
        return r

    def matmul(self, x: Tensor, y: Tensor) -> Tensor:
        r = torch.matmul(x, y)
        r = self.activation_post_process(r)
        return r

    @classmethod
    def from_float(cls, float_module):
        assert hasattr(float_module, 'qconfig')
        observed = cls()
        observed.qconfig = float_module.qconfig
        return observed


class ExtendedQuantizedOps(torch.nn.Module):
    def __init__(self):
        super(ExtendedQuantizedOps, self).__init__()
        self.scale = 1.0
        self.zero_point = 0
        self.activation_post_process = torch.nn.Identity()

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        super(ExtendedQuantizedOps, self)._save_to_state_dict(destination, prefix, keep_vars)
        destination[prefix + 'scale'] = torch.tensor(self.scale)
        destination[prefix + 'zero_point'] = torch.tensor(self.zero_point)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):

        self.scale = float(state_dict.pop(prefix + 'scale'))
        self.zero_point = int(state_dict.pop(prefix + 'zero_point'))
        super(ExtendedQuantizedOps, self)._load_from_state_dict(state_dict, prefix, local_metadata, False,
                                                       missing_keys, unexpected_keys, error_msgs)

    def _get_name(self):
        return 'ExtendedQuantizedOps'

    def extra_repr(self):
        return 'scale={}, zero_point={}'.format(
            self.scale, self.zero_point
        )

    def forward(self, x):
        raise RuntimeError("Functional is not intended to use the " +
                           "'forward'. Please use the underlying operation")

    def softmax(self, x: Tensor, dim=-1) -> Tensor:
        r = torch.ops.quantized.softmax(x, dim, self.scale, self.zero_point)
        r = self.activation_post_process(r)
        return r

    def matmul(self, x: Tensor, y: Tensor) -> Tensor:
        IPython.embed()
        r = torch.ops.quantized.matmul(x, y, self.scale, self.zero_point)
        r = self.activation_post_process(r)
        return r

    @classmethod
    def from_observed(cls, mod):
        assert hasattr(mod, 'qconfig')
        assert hasattr(mod, 'activation_post_process')
        assert type(mod) == ExtendedQuantizedOpsObserver,\
            "ExtendedQuantizedObservers.from_float expects an instance of ExtendedQuantizedObservers"
        scale, zero_point = mod.activation_post_process.calculate_qparams()  # type: ignore[operator]
        new_mod = ExtendedQuantizedOps()
        new_mod.scale = float(scale)
        new_mod.zero_point = int(zero_point)
        return new_mod
