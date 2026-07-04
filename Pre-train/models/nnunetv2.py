# Copyright (c) ByteDance, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
from monai.networks.blocks.convolutions import Convolution, ResidualUnit
from monai.networks.layers.factories import Act, Norm
from timm.models.layers import trunc_normal_
from timm.models.registry import register_model
from torch.nn import functional as F


class AttentionPool3d(nn.Module):
    def __init__(self, 
                 voxel_counts: int, 
                 embed_dim: int, 
                 num_heads: int, 
                 output_dim: int = None):
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(voxel_counts + 1, embed_dim) / embed_dim ** 0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x):
        x = x.flatten(start_dim=2).permute(2, 0, 1)  # NCHWD -> (HWD)NC
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HWD+1)NC
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (HWD+1)NC
        x, _ = F.multi_head_attention_forward(
            query=x[:1], key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False
        )
        return x.squeeze(0)


class CCIPEncoder(nn.Module):
    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        strides: Sequence[int],
        kernel_size: Union[Sequence[int], int] = 3,
        up_kernel_size: Union[Sequence[int], int] = 3,
        num_res_units: int = 0,
        act: Union[Tuple, str] = Act.PRELU,
        norm: Union[Tuple, str] = Norm.INSTANCE,
        dropout: float = 0.0,
        bias: bool = False,
        adn_ordering: str = "NDA",
        dimensions: Optional[int] = None,
    ) -> None:

        super().__init__()

        if isinstance(kernel_size, Sequence):
            if len(kernel_size) != spatial_dims:
                raise ValueError("the length of `kernel_size` should equal to `dimensions`.")
        if isinstance(up_kernel_size, Sequence):
            if len(up_kernel_size) != spatial_dims:
                raise ValueError("the length of `up_kernel_size` should equal to `dimensions`.")

        self.dimensions = spatial_dims
        self.in_channels = in_channels
        self.strides = strides
        self.kernel_size = kernel_size
        self.up_kernel_size = up_kernel_size
        self.num_res_units = num_res_units
        self.act = act
        self.norm = norm
        self.dropout = dropout
        self.bias = bias
        self.adn_ordering = adn_ordering
        self.image_size = None

        self.c11 = self._get_bottom_layer(in_channels, 32)
        self.c12 = self._get_bottom_layer(32, 32)
        self.c21 = self._get_down_layer(32, 64, strides[0], False)
        self.c22 = self._get_bottom_layer(64, 64)
        self.c31 = self._get_down_layer(64, 128, strides[1], False)
        self.c32 = self._get_bottom_layer(128, 128)
        self.c41 = self._get_down_layer(128, 256, strides[2], False)
        self.c42 = self._get_bottom_layer(256, 256)
        self.c51 = self._get_down_layer(256, 320, strides[3], False)
        self.c52 = self._get_bottom_layer(320, 320)

        # self.attnpool = AttentionPool3d(9*11*9, 320, 16, 256)

        # self.initialize_parameters()

    def initialize_parameters(self):
        if self.attnpool is not None:
            std = self.attnpool.c_proj.in_features ** -0.5
            nn.init.normal_(self.attnpool.q_proj.weight, std=std)
            nn.init.normal_(self.attnpool.k_proj.weight, std=std)
            nn.init.normal_(self.attnpool.v_proj.weight, std=std)
            nn.init.normal_(self.attnpool.c_proj.weight, std=std)

    def _get_down_layer(self, in_channels: int, out_channels: int, strides: int, is_top: bool) -> nn.Module:
        """
        Returns the encoding (down) part of a layer of the network. This typically will downsample data at some point
        in its structure. Its output is used as input to the next layer down and is concatenated with output from the
        next layer to form the input for the decode (up) part of the layer.

        Args:
            in_channels: number of input channels.
            out_channels: number of output channels.
            strides: convolution stride.
            is_top: True if this is the top block.
        """
        mod: nn.Module
        if self.num_res_units > 0:

            mod = ResidualUnit(
                self.dimensions,
                in_channels,
                out_channels,
                strides=strides,
                kernel_size=self.kernel_size,
                subunits=self.num_res_units,
                act=self.act,
                norm=self.norm,
                dropout=self.dropout,
                bias=self.bias,
                adn_ordering=self.adn_ordering,
            )
            return mod
        mod = Convolution(
            self.dimensions,
            in_channels,
            out_channels,
            strides=strides,
            kernel_size=self.kernel_size,
            act=self.act,
            norm=self.norm,
            dropout=self.dropout,
            bias=self.bias,
            adn_ordering=self.adn_ordering
        )
        return mod

    def _get_bottom_layer(self, in_channels: int, out_channels: int) -> nn.Module:
        """
        Returns the bottom or bottleneck layer at the bottom of the network linking encode to decode halves.

        Args:
            in_channels: number of input channels.
            out_channels: number of output channels.
        """
        return self._get_down_layer(in_channels, out_channels, 1, False)

    def _get_up_layer(self, in_channels: int, out_channels: int, strides: int, is_top: bool) -> nn.Module:
        """
        Returns the decoding (up) part of a layer of the network. This typically will upsample data at some point
        in its structure. Its output is used as input to the next layer up.

        Args:
            in_channels: number of input channels.
            out_channels: number of output channels.
            strides: convolution stride.
            is_top: True if this is the top block.
        """
        conv: Union[Convolution, nn.Sequential]

        conv = Convolution(
            self.dimensions,
            in_channels,
            out_channels,
            strides=strides,
            kernel_size=self.up_kernel_size,
            bias=True,
            conv_only=True,
            is_transposed=True,
            adn_ordering=self.adn_ordering,
        )

        if self.num_res_units > 0:
            ru = ResidualUnit(
                self.dimensions,
                out_channels,
                out_channels,
                strides=1,
                kernel_size=self.kernel_size,
                subunits=1,
                act=self.act,
                norm=self.norm,
                dropout=self.dropout,
                bias=self.bias,
                last_conv_only=is_top,
                adn_ordering=self.adn_ordering,
            )
            conv = nn.Sequential(conv, ru)

        return conv

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x11 = self.c11(x)
        x12 = self.c12(x11)
        x21 = self.c21(x12)
        x22 = self.c22(x21)
        x31 = self.c31(x22)
        x32 = self.c32(x31)
        x41 = self.c41(x32)
        x42 = self.c42(x41)
        x51 = self.c51(x42)
        x52 = self.c52(x51)
        # output = self.attnpool(x52)
        return x52


class UNetEncoder(nn.Module):
    """
    This is a template for your custom ConvNet.
    It is required to implement the following three functions: `get_downsample_ratio`, `get_feature_map_channels`, `forward`.
    You can refer to the implementations in `pretrain\models\resnet.py` for an example.
    """
    def __init__(self, 
                 in_chans=1, 
                 depths=[2, 2, 2, 2, 1], 
                 dims=[32, 64, 128, 256, 320],
                 sparse=True,
                 ):
        super().__init__()

        self.stem = self._get_bottom_layer(in_chans, dims[0])
        self.stages = nn.ModuleList() 
        self.downsample_layers = nn.ModuleList()
        self.n_stages = len(depths)
        self.dims = dims
        
        for i in range(len(depths)):
            stage = nn.Sequential(
                *[self._get_bottom_layer(dims[i], dims[i]) for j in range(depths[i]-1)],
            )
            if i != len(depths) - 1:
                self.downsample_layers.append(self._get_down_layer(dims[i], dims[i+1]))

            self.stages.append(stage)

        # self.apply(self._init_weights)

    def _get_bottom_layer(self, in_channels, out_channels):
        return nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.InstanceNorm3d(out_channels),
                nn.PReLU(),
        )

    def _get_down_layer(self, in_channels, out_channels):
        return nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
                nn.InstanceNorm3d(out_channels),
                nn.PReLU(),
        )

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv3d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)
    
    def get_downsample_ratio(self) -> int:
        """
        This func would ONLY be used in `SparseEncoder's __init__` (see `pretrain/encoder.py`).
        
        :return: the TOTAL downsample ratio of the ConvNet.
        E.g., for a ResNet-50, this should return 32.
        """
        return 2 ** (self.n_stages - 1)
    
    def get_feature_map_channels(self) -> List[int]:
        """
        This func would ONLY be used in `SparseEncoder's __init__` (see `pretrain/encoder.py`).
        
        :return: a list of the number of channels of each feature map.
        E.g., for a ResNet-50, this should return [256, 512, 1024, 2048].
        """
        return self.dims
    
    def forward(self, inp_bchwd: torch.Tensor, hierarchical=True):
        """
        The forward with `hierarchical=True` would ONLY be used in `SparseEncoder.forward` (see `pretrain/encoder.py`).
        
        :param inp_bchw: input image tensor, shape: (batch_size, channels, height, width).
        :param hierarchical: return the logits (not hierarchical), or the feature maps (hierarchical).
        :return:
            - hierarchical == False: return the logits of the classification task, shape: (batch_size, num_classes).
            - hierarchical == True: return a list of all feature maps, which should have the same length as the return value of `get_feature_map_channels`.
              E.g., for a ResNet-50, it should return a list [1st_feat_map, 2nd_feat_map, 3rd_feat_map, 4th_feat_map].
                    for an input size of 224, the shapes are [(B, 256, 56, 56), (B, 512, 28, 28), (B, 1024, 14, 14), (B, 2048, 7, 7)]
        """
        if hierarchical:
            x = self.stem(inp_bchwd)
            ls = []
            for i in range(self.n_stages):
                x = self.stages[i](x)             
                ls.append(x)
                if i != self.n_stages - 1:
                    x = self.downsample_layers[i](x)
            return ls
        else:
            raise NotImplementedError


@register_model
def build_unet_decoder(pretrained=False, **kwargs):
    return UNetEncoder(**kwargs)


@torch.no_grad()
def convnet_test():
    cnn = UNetEncoder().cuda()
    print('get_downsample_ratio:', cnn.get_downsample_ratio())
    print('get_feature_map_channels:', cnn.get_feature_map_channels())
    
    downsample_ratio = cnn.get_downsample_ratio()
    feature_map_channels = cnn.get_feature_map_channels()
    
    # check the forward function
    B, C, H, W, D = 4, 1, 96, 96, 96
    inp = torch.rand(B, C, H, W, D).cuda()
    feats = cnn(inp, hierarchical=True)
    assert isinstance(feats, list)
    assert len(feats) == len(feature_map_channels)
    print([tuple(t.shape) for t in feats])
    
    # check the downsample ratio
    feats = cnn(inp, hierarchical=True)
    assert feats[-1].shape[-2] == H // downsample_ratio
    assert feats[-1].shape[-1] == W // downsample_ratio
    
    # check the channel number
    for feat, ch in zip(feats, feature_map_channels):
        assert feat.ndim == 5
        assert feat.shape[1] == ch

class UNetDecoder(nn.Module):
    def __init__(self, 
                 out_chans=1, 
                 depths=[2, 2, 2, 2, 0], 
                 dims=[32, 64, 128, 256, 320],
                 ):
        super().__init__()

        self.stages = nn.ModuleList() 
        self.upsample_layers = nn.ModuleList()
        self.n_stages = len(depths)
        self.dims = dims
        self.width = dims
        
        for i in range(len(depths)-1, -1, -1):
            if depths[i] > 0:
                stage = nn.Sequential(
                    self._get_bottom_layer(dims[i] * 2, dims[i]),
                    *[self._get_bottom_layer(dims[i], dims[i]) for j in range(depths[i]-1)],
                )
                self.stages.append(stage)

            if i != 0:
                self.upsample_layers.append(self._get_up_layer(dims[i], dims[i-1]))   

        # self.proj = nn.Conv3d(dims[0], out_chans, kernel_size=1, stride=1, bias=True)
        # self.proj.weight = nn.Parameter(Normal(0, 1e-5).sample(self.proj.weight.shape))
        # self.proj.bias = nn.Parameter(torch.zeros(self.proj.bias.shape))

    def _get_up_layer(self, in_channels, out_channels):
        return nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2, bias=False)

    def _get_bottom_layer(self, in_channels, out_channels):
        return nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.InstanceNorm3d(out_channels),
                nn.PReLU(),
        )
    
    def forward(self, to_dec: List[torch.Tensor]):
        x = to_dec[0]
        for i, d in enumerate(self.stages):
            x = self.upsample_layers[i](x)
            if i + 1 < len(to_dec):
                x = torch.cat([x, to_dec[i+1]], dim=1)
            x = d(x)

        # return self.proj(x)
        return x
    

if __name__ == '__main__':
    convnet_test()