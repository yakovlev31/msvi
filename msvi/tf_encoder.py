import torch
import torch.nn as nn


Tensor = torch.Tensor
Module = nn.Module


class TFEncoder(nn.Module):
    # Modified https://pytorch.org/docs/stable/_modules/torch/nn/modules/transformer.html#TransformerEncoderLayer
    def __init__(
        self,
        d_model: int,
        self_attn: Module,
        t: Tensor,
        dim_feedforward: int = 2048,
        dropout: float = 0.0,
        layer_norm_eps: float = 1e-5,
        **kwargs,
    ) -> None:

        super().__init__()

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm([d_model], eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm([d_model], eps=layer_norm_eps)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = torch.nn.functional.relu  # type: ignore
        self.self_attn = self_attn

    def forward(self, x: Tensor) -> Tensor:
        x = self.norm1(x + self._sa_block(x))
        x = self.norm2(x + self._ff_block(x))
        return x

    # self-attention block
    def _sa_block(self, x: Tensor) -> Tensor:
        x = self.self_attn(x, return_weights=False)[0]
        return self.dropout1(x)

    # feed forward block
    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)
