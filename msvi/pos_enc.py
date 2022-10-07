from typing import Union
import numpy as np
import torch
import torch.nn as nn


Tensor = torch.Tensor
Module = nn.Module
Sequential = nn.Sequential


class DiscreteSinCosPositionalEncoding(Module):
    # Modified https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    def __init__(self, d_model: int, t: Tensor, max_tokens: int, dropout: float = 0.0, **kwargs):
        assert d_model % 2 == 0, "d_model must be even."
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        self.d_model = d_model
        self.max_tokens = max_tokens

        self.update_time_grid(t)

    def forward(self, x: Tensor) -> Tensor:
        # x: Tensor, shape (S, M, K).
        x = x + self.pe
        return self.dropout(x)

    def update_time_grid(self, t: Tensor) -> None:
        # t: Tensor, shape (S, M, 1).
        # assert torch.all((t - t[0]) < 1e-7).item() is True, "All time grids must be the same."
        _, M, _ = t.shape
        position = torch.arange(M, device=t.device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.d_model, 2, device=t.device) * (-np.log(self.max_tokens) / self.d_model))
        pe = torch.zeros(1, M, self.d_model, device=t.device)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.pe = pe


class ContinuousSinCosPositionalEncoding(Module):
    # Modified https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    def __init__(self, d_model: int, t: Tensor, max_tokens: int, max_time: float, dropout: float = 0.0, **kwargs):
        assert d_model % 2 == 0, "d_model must be even."
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        self.d_model = d_model
        self.max_tokens = max_tokens
        self.max_time = max_time

        self.update_time_grid(t)

    def forward(self, x: Tensor) -> Tensor:
        # x: Tensor, shape (S, M, K).
        x = x + self.pe
        return self.dropout(x)

    def update_time_grid(self, t: Tensor) -> None:
        # t: Tensor, shape (S, M, 1).
        S, M, _ = t.shape
        position = t / self.max_time * (self.max_tokens - 1)
        div_term = torch.exp(torch.arange(0, self.d_model, 2, device=t.device) * (-np.log(self.max_tokens) / self.d_model))  # (K/2,)
        pe = torch.zeros(S, M, self.d_model, device=t.device)
        pe[:, :, 0::2] = torch.sin(position * div_term)
        pe[:, :, 1::2] = torch.cos(position * div_term)
        self.pe = pe


class RelativePositionalEncodingNN(Module):
    def __init__(self, f: Union[Module, Sequential], t: Tensor, delta_r: float, **kwargs):
        super().__init__()

        self.f = f
        self.delta_r = delta_r
        self.squish_fn = nn.Hardtanh()

        self.update_time_grid(t)

    def forward(self) -> Tensor:
        rpe = self.f(self.dt_prime_mat)
        return rpe

    def update_time_grid(self, t: Tensor) -> None:
        # t: Tensor, shape (S, M, 1).
        dt_mat = self._get_dt_matrix(t)
        self.dt_prime_mat = self.squish_fn(dt_mat / self.delta_r).float()

    def _get_dt_matrix(self, t: Tensor) -> Tensor:
        """Calculates the matrix of relative distances between all time points in `t`."""
        dist_mat = torch.cdist(t, t, p=1)  # (S, M, M)
        dir_mat = torch.ones_like(dist_mat).triu() - torch.ones_like(dist_mat).tril()  # (S, M, M)
        dt_mat = (dir_mat * dist_mat).unsqueeze(-1)  # (S, M, M, 1)
        return dt_mat


class RelativePositionalEncodingInterp(Module):
    def __init__(self, d_model: int, t: Tensor, delta_r: float, **kwargs):
        super().__init__()

        self.delta_r = delta_r
        self.squish_fn = nn.Hardtanh()

        self._set_random_vectors(d_model)
        self.update_time_grid(t)

    def forward(self) -> Tensor:
        return self.pe

    def update_time_grid(self, t: Tensor) -> None:
        # t: Tensor, shape (S, M, 1).
        dt_mat = self._get_dt_matrix(t)
        dt_prime_mat = self.squish_fn(dt_mat / self.delta_r).float()

        self.lm = (dt_prime_mat + 1) / 2
        pe = ((1 - self.lm) * self.va + self.lm * self.vb)

        self.pe = pe

    def _set_random_vectors(self, d_model: int) -> None:
        va_ = (torch.rand((1, d_model)) - 0.5) * 2
        va = va_ / torch.linalg.norm(va_, ord=np.inf)
        vb = -va
        self.register_buffer("va", va)
        self.register_buffer("vb", vb)

    def _get_dt_matrix(self, t: Tensor) -> Tensor:
        """Calculates the matrix of relative distances between all time points in `t`."""
        dist_mat = torch.cdist(t, t, p=1)  # (S, M, M)
        dir_mat = torch.ones_like(dist_mat).triu() - torch.ones_like(dist_mat).tril()  # (S, M, M)
        dt_mat = (dir_mat * dist_mat).unsqueeze(-1)  # (S, M, M, 1)
        return dt_mat
