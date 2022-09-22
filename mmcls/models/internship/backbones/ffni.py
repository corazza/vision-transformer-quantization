from mmcv.cnn.bricks.registry import FEEDFORWARD_NETWORK
from copy import deepcopy
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.utils.checkpoint as cp
from mmcv.cnn import build_norm_layer
from mmcv.cnn.bricks.transformer import PatchEmbed, AdaptivePadding #, PatchMerging
from mmcv.cnn.utils.weight_init import trunc_normal_
from mmcv.runner.base_module import BaseModule, ModuleList
from mmcv.utils.parrots_wrapper import _BatchNorm

from mmcv.cnn.bricks.transformer import FFN
from ...builder import BACKBONES
from ...utils import (ShiftWindowMSA, resize_pos_embed,
                     resize_relative_position_bias_table, to_2tuple)
from ...backbones.base_backbone import BaseBackbone
import torch.nn.quantized
from mmcv.cnn import (Linear, build_activation_layer, build_conv_layer,
                      build_norm_layer)
from mmcv.runner.base_module import BaseModule, ModuleList, Sequential
from mmcv.cnn.bricks.drop import build_dropout
from mmcv.utils import (ConfigDict, build_from_cfg, deprecated_api_warning,
                        to_2tuple)


@FEEDFORWARD_NETWORK.register_module()
class FFNI(BaseModule):
    """Implements feed-forward networks (FFNs) with identity connection.

    Args:
        embed_dims (int): The feature dimension. Same as
            `MultiheadAttention`. Defaults: 256.
        feedforward_channels (int): The hidden dimension of FFNs.
            Defaults: 1024.
        num_fcs (int, optional): The number of fully-connected layers in
            FFNs. Default: 2.
        act_cfg (dict, optional): The activation config for FFNs.
            Default: dict(type='ReLU')
        ffn_drop (float, optional): Probability of an element to be
            zeroed in FFN. Default 0.0.
        add_identity (bool, optional): Whether to add the
            identity connection. Default: `True`.
        dropout_layer (obj:`ConfigDict`): The dropout_layer used
            when adding the shortcut.
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Default: None.
    """
    @deprecated_api_warning(
        {
            'dropout': 'ffn_drop',
            'add_residual': 'add_identity'
        },
        cls_name='FFNI')
    def __init__(self,
                 embed_dims=256,
                 feedforward_channels=1024,
                 num_fcs=2,
                 act_cfg=dict(type='ReLU', inplace=True),
                 ffn_drop=0.,
                 dropout_layer=None,
                 add_identity=True,
                 init_cfg=None,
                 overwrite_act_with_relu=True,
                 **kwargs):
        super().__init__(init_cfg)
        self.overwrite_act_with_relu = overwrite_act_with_relu
        if self.overwrite_act_with_relu:
            act_cfg=dict(type='ReLU', inplace=True)
        assert num_fcs == 2, 'num_fcs shoGld be ' \
            f' 2. got {num_fcs}.'
        self.quant1 = torch.quantization.QuantStub()
        self.quant2 = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()
        self.f_add = torch.nn.quantized.FloatFunctional()
        self.embed_dims = embed_dims
        self.feedforward_channels = feedforward_channels
        self.num_fcs = num_fcs
        self.act_cfg = act_cfg
        self.activate = build_activation_layer(act_cfg)

        layers = []
        in_channels = embed_dims
        for _ in range(num_fcs - 1):
            layer = Sequential(
                    nn.Linear(in_channels, feedforward_channels), self.activate,
                    # Linear(in_channels, feedforward_channels), self.activate,
                    nn.Dropout(ffn_drop))
            layers.append(layer)
            in_channels = feedforward_channels
        layer = nn.Linear(feedforward_channels, embed_dims)
        # layer = Linear(feedforward_channels, embed_dims)
        layers.append(layer)
        layers.append(nn.Dropout(ffn_drop))
        self.layers = Sequential(*layers)

        self.dropout_layer = build_dropout(
            dropout_layer) if dropout_layer else torch.nn.Identity()
        self.add_identity = add_identity

    def insert_observers(self):
        # if not self.overwrite_act_with_relu:
        #     self.activate.qconfig = None # Needed for skipping GeLU quantization
        # self.activate.qconfig = None # Needed for skipping GeLU quantization
        self.layers = torch.quantization.add_quant_dequant(self.layers)

    @deprecated_api_warning({'residual': 'identity'}, cls_name='FFNI')
    def forward(self, x, identity=None):
        """Forward function for `FFN`.

        The function would add x to the output tensor if residue is None.
        """
        # x = self.quant1(x)
        out = self.layers(x)
        out = self.quant1(out)
        if not self.add_identity:
            return self.dequant(self.dropout_layer(out))
        if identity is None:
            identity = self.quant1(x)
            # identity = x
        else:
            identity = self.quant2(identity)
            # identity = identity
        # return identity + self.dropout_layer(out)
        return self.dequant(self.f_add.add(identity, self.dropout_layer(out)))
