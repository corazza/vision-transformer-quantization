import torch
from torch import Tensor
from torch._ops import ops
from torch.nn import functional as F

import IPython


class ExtendedQuantizedObservers(torch.nn.Module):
    def __init__(self, dim=-1):
        super(ExtendedQuantizedObservers, self).__init__()
        self.dim = dim

    def forward(self, x):
        raise RuntimeError("FloatFunctional is not intended to use the " +
                           "'forward'. Please use the underlying operation")

    def softmax(self, x: Tensor) -> Tensor:
        r = F.softmax(x, dim=self.dim)
        r = self.activation_post_process(r)
        return r # TODO FIXME observer -> quantizer doesn't carry over dim information, had to hardcode

    @classmethod
    def from_float(cls, float_module):
        assert hasattr(float_module, 'qconfig')
        observed = cls()
        observed.qconfig = float_module.qconfig
        return observed


class ExtendedQuantizedOps(torch.nn.Module):
    def __init__(self, dim):
        super(ExtendedQuantizedOps, self).__init__()
        self.dim = dim
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

    def softmax(self, x: Tensor) -> Tensor:
        r = torch.ops.quantized.softmax(x, self.dim, self.scale, self.zero_point)
        r = self.activation_post_process(r)
        return r

    @classmethod
    def from_observed(cls, mod):
        assert hasattr(mod, 'qconfig')
        assert hasattr(mod, 'activation_post_process')
        assert type(mod) == ExtendedQuantizedObservers,\
            "ExtendedQuantizedObservers.from_float expects an instance of ExtendedQuantizedObservers"
        scale, zero_point = mod.activation_post_process.calculate_qparams()  # type: ignore[operator]
        new_mod = ExtendedQuantizedOps(mod.dim)
        new_mod.scale = float(scale)
        new_mod.zero_point = int(zero_point)
        return new_mod
