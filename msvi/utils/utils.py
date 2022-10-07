from types import SimpleNamespace

import torch
import torch.nn as nn

from einops import rearrange

from msvi.pos_enc import (
    DiscreteSinCosPositionalEncoding,
    ContinuousSinCosPositionalEncoding,
    RelativePositionalEncodingInterp,
    RelativePositionalEncodingNN
)

from msvi.attention import (
    DotProductAttention,
    TemporalAttention,
    TemporalDotProductAttention,
    TemporalDotProductAttentionBaseline,
)

from msvi.tf_encoder import TFEncoder


Tensor = torch.Tensor
Module = torch.nn.Module
Sequential = torch.nn.Sequential


def create_agg_net(param: SimpleNamespace, net_type: str) -> Sequential:
    """Constructs aggregation network."""

    pos_enc_layers = {
        "dsc": DiscreteSinCosPositionalEncoding,
        "csc": ContinuousSinCosPositionalEncoding,
        "rpeNN": RelativePositionalEncodingNN,
        "rpeInterp": RelativePositionalEncodingInterp,
        "none": None,
    }

    attn_layers = {
        "dp": DotProductAttention,
        "t": TemporalAttention,
        "tdp": TemporalDotProductAttention,
        "tdp_b": TemporalDotProductAttentionBaseline,
    }

    attn_key, pos_enc_key = param.h_agg_attn, param.h_agg_pos_enc
    assert pos_enc_key in pos_enc_layers.keys(), f"Wrong position encoding name: {pos_enc_key}."
    assert attn_key in attn_layers.keys(), f"Wrong attention layer name: {attn_key}."

    t_init = torch.linspace(0, 1, 3).view(1, -1, 1)  # update it later
    pos_enc_args = {
        "d_model": param.m_h*param.K,
        "t": t_init,
        "max_tokens": param.h_agg_max_tokens,
        "max_time": param.h_agg_max_time,
        "delta_r": param.h_agg_delta_r,
        "f": nn.Linear(1, param.m_h*param.K, bias=False),
    }
    attn_args = {
        "d_model": param.m_h*param.K,
        "t": t_init,
        "eps": 1e-2,
        "delta_r": param.h_agg_delta_r,
        "p": param.h_agg_p,
        "n": param.n,
        "drop_prob": param.drop_prob,
    }

    if net_type == "static":
        param.h_agg_layers = param.h_agg_stat_layers
    elif net_type == "dynamic":
        param.h_agg_layers = param.h_agg_dyn_layers

    modules = []
    if pos_enc_key in ["dsc", "csc"]:  # absolute positional encodings
        pos_enc = pos_enc_layers[pos_enc_key](**pos_enc_args)
        tf_enc_blocks = []
        for _ in range(param.h_agg_layers):
            tf_enc_block = TFEncoder(
                d_model=param.m_h*param.K,
                self_attn=attn_layers[attn_key](**attn_args),
                t=t_init,
                dim_feedforward=2*param.m_h*param.K,
            )
            tf_enc_blocks.append(tf_enc_block)
        modules.extend([pos_enc, *tf_enc_blocks])
    else:  # relative positional encodings
        if pos_enc_key == "none":
            print("Using no positional encodings!")
            pos_enc = None
        else:
            pos_enc = pos_enc_layers[pos_enc_key](**pos_enc_args)
        tf_enc_blocks = []
        for i in range(param.h_agg_layers):
            if i == 0:
                self_attn = attn_layers["t"](rpe=pos_enc, **attn_args)
                # self_attn = attn_layers[attn_key](rpe=pos_enc_layers[pos_enc_key](**pos_enc_args), **attn_args)
            else:
                self_attn = attn_layers[attn_key](rpe=pos_enc, **attn_args)

            tf_enc_block = TFEncoder(
                d_model=param.m_h*param.K,
                self_attn=self_attn,
                t=t_init,
                dim_feedforward=2*param.m_h*param.K,
            )

            # if i != (param.h_agg_layers - 1):
            #     print(f"Layer {i}, adding dropout")
            #     tf_enc_block = TFEncoder(
            #         d_model=param.m_h*param.K,
            #         self_attn=self_attn,
            #         t=t_init,
            #         dim_feedforward=2*param.m_h*param.K,
            #         dropout=0.1,
            #     )
            # else:
            #     print(f"Layer {i}, no dropout")
            #     tf_enc_block = TFEncoder(
            #         d_model=param.m_h*param.K,
            #         self_attn=self_attn,
            #         t=t_init,
            #         dim_feedforward=2*param.m_h*param.K,
            #     )

            tf_enc_blocks.append(tf_enc_block)
        modules.extend(tf_enc_blocks)

    return nn.Sequential(*modules)


class CNNEncoder(Module):
    """Mapping from R^{NxD} to R^K."""
    def __init__(self, K: int, N: int, D: int, n_channels: int) -> None:
        super().__init__()

        self.K = K
        self.N = N
        self.D = D

        self.n_channels = n_channels
        self.img_size = int(N**0.5)
        self.n_feat = (self.img_size//16)**2 * (8 * n_channels)

        self.f = nn.Sequential(
            nn.Conv2d(D, n_channels, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.BatchNorm2d(n_channels),  # img_size/2

            nn.Conv2d(n_channels, 2*n_channels, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.BatchNorm2d(2*n_channels),  # img_size/4

            nn.Conv2d(2*n_channels, 4*n_channels, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.BatchNorm2d(4*n_channels),  # img_size/8

            nn.Conv2d(4*n_channels, 8*n_channels, kernel_size=2, stride=2, padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(8*n_channels),  # img_size/16

            nn.Flatten(),
            nn.Linear(self.n_feat, K),
        )

    def forward(self, x: Tensor) -> Tensor:
        # x: Tensor, shape (S, M, N, D)
        S, M, _, _ = x.shape
        x = rearrange(x, "s m (h w) d -> (s m) d h w", h=self.img_size, w=self.img_size)
        x = self.f(x)
        x = rearrange(x, "(s m) k -> s m k", s=S, m=M)
        return x


class CNNDecoder(Module):
    """Mapping from R^K to R^{NxDxn_param}."""
    def __init__(self, K: int, N: int, D: int, n_param: int, n_channels: int) -> None:
        super().__init__()

        self.K = K
        self.N = N
        self.D = D
        self.n_param = n_param

        self.n_channels = n_channels
        self.img_size = int(N**0.5)
        self.n_feat = (self.img_size//16)**2 * (8 * n_channels)

        self.lin_layer = nn.Linear(K, self.n_feat)

        self.f = nn.Sequential(
            nn.ConvTranspose2d(8*n_channels, 4*n_channels, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.BatchNorm2d(4*n_channels),  # img_size/8

            nn.ConvTranspose2d(4*n_channels, 2*n_channels, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.BatchNorm2d(2*n_channels),  # img_size/4

            nn.ConvTranspose2d(2*n_channels, n_channels, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.BatchNorm2d(n_channels),  # img_size/2

            nn.ConvTranspose2d(n_channels, n_channels, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.BatchNorm2d(n_channels),  # img_size

            nn.Conv2d(n_channels, D*n_param, kernel_size=5, padding=2),
        )

    def forward(self, x: Tensor) -> Tensor:
        # x: Tensor, shape (S, M, K)
        S, M, _ = x.shape
        nc, h = 8*self.n_channels, self.img_size//16
        x = rearrange(self.lin_layer(x), "s m (nc h w) -> (s m) nc h w", nc=nc, h=h, w=h)
        x = self.f(x)
        x = rearrange(x, "(s m) (d npar) h w -> s m (h w) d npar", s=S, m=M, d=self.D, npar=self.n_param)
        return x


class Sine(nn.Module):
    def __init__(self, w=1.0):
        super().__init__()
        self.weight = nn.parameter.Parameter(torch.tensor(w), True)
        self.bias = nn.parameter.Parameter(torch.tensor(0.0), False)

    def forward(self, x):
        return torch.sin(self.weight * x)


class CNNEncoderScalarFlow(Module):
    """Mapping from R^{NxD} to R^K."""
    def __init__(self, K: int, N: int, D: int, n_channels: int) -> None:
        super().__init__()

        self.K = K
        self.N = N
        self.D = D

        self.n_channels = n_channels
        self.n_feat = 8 * (8 * n_channels)

        self.f = nn.Sequential(
            nn.Conv2d(D, n_channels, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.BatchNorm2d(n_channels),  # img_size/2

            nn.Conv2d(n_channels, 2*n_channels, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.BatchNorm2d(2*n_channels),  # img_size/4

            nn.Conv2d(2*n_channels, 4*n_channels, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.BatchNorm2d(4*n_channels),  # img_size/8

            nn.Conv2d(4*n_channels, 8*n_channels, kernel_size=2, stride=2, padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(8*n_channels),  # img_size/16

            nn.Flatten(),
            nn.Linear(self.n_feat, K),
        )

    def forward(self, x: Tensor) -> Tensor:
        # x: Tensor, shape (S, M, N, D)
        S, M, _, _ = x.shape
        x = rearrange(x, "s m (h w) d -> (s m) d h w", h=64, w=32)
        x = self.f(x)
        x = rearrange(x, "(s m) k -> s m k", s=S, m=M)
        return x


class CNNDecoderScalarFlow(Module):
    """Mapping from R^K to R^{NxDxn_param}."""
    def __init__(self, K: int, N: int, D: int, n_param: int, n_channels: int) -> None:
        super().__init__()

        self.K = K
        self.N = N
        self.D = D
        self.n_param = n_param

        self.n_channels = n_channels
        self.n_feat = 8 * (8 * n_channels)

        self.lin_layer = nn.Linear(K, self.n_feat)

        self.f = nn.Sequential(
            nn.ConvTranspose2d(8*n_channels, 4*n_channels, kernel_size=(4, 2), stride=(2, 2), padding=(1, 0)),
            nn.ReLU(),
            nn.BatchNorm2d(4*n_channels),  # img_size/8

            nn.ConvTranspose2d(4*n_channels, 2*n_channels, kernel_size=(8, 4), stride=(2, 2), padding=(3, 1)),
            nn.ReLU(),
            nn.BatchNorm2d(2*n_channels),  # img_size/4

            nn.ConvTranspose2d(2*n_channels, n_channels, kernel_size=(8, 4), stride=(2, 2), padding=(3, 1)),
            nn.ReLU(),
            nn.BatchNorm2d(n_channels),  # img_size/2

            nn.ConvTranspose2d(n_channels, n_channels, kernel_size=(8, 4), stride=(2, 2), padding=(3, 1)),
            nn.ReLU(),
            nn.BatchNorm2d(n_channels),  # img_size

            nn.Conv2d(n_channels, D*n_param, kernel_size=11, padding=5),
        )

    def forward(self, x: Tensor) -> Tensor:
        # x: Tensor, shape (S, M, K)
        S, M, _ = x.shape
        nc = 8 * self.n_channels
        x = rearrange(self.lin_layer(x), "s m (nc h w) -> (s m) nc h w", nc=nc, h=4, w=2)
        x = self.f(x)
        x = rearrange(x, "(s m) (d npar) h w -> s m (h w) d npar", s=S, m=M, d=self.D, npar=self.n_param)
        return x
