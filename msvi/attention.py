from abc import ABC, abstractmethod

from typing import Union

import numpy as np
import torch
import torch.nn as nn


Tensor = torch.Tensor
Module = nn.Module


class IAttention(Module, ABC):
    @abstractmethod
    def forward(
        self,
        x: Tensor,
        return_weights: bool = True
    ) -> Union[tuple[Tensor, Tensor], tuple[Tensor, None]]:

        """Maps input sequence x to output sequence.

        Args:
            x: Input sequence. Has shape (S, M, K).
            return_weights: If True, returns attention weights. Otherwise, returns None.

        Returns:
            y: Output sequence. Has shape (S, M, K).
            attn_weights: Attention weights. Has shape (S, M, M).
                None is returned if `return_weights` is False.
        """
        pass

    @abstractmethod
    def update_time_grid(self, t: Tensor) -> None:
        """Updates all parts of the class that depend on time grids (except submodules
            which might also depend on time grids, those must be upated separately
            (see msvi.rec_net)).

        Args:
            t: New time grids. Has shape (S, M, 1).
        """
        pass


class AttentionBase(IAttention):
    def __init__(self, d_model: int, rpe: Union[Module, None] = None, drop_prob: float = 0.0):
        super().__init__()
        self.d_model = d_model
        self.rpe = rpe
        self.drop_prob = drop_prob

    def forward(self, x: Tensor, return_weights: bool = True) -> Union[tuple[Tensor, Tensor], tuple[Tensor, None]]:
        attn_weights = self._eval_attn_weights(x)
        output = self._eval_output(attn_weights, x)
        if return_weights:
            return output, attn_weights
        else:
            return output, None

    def drop(self, w: Tensor) -> Tensor:
        """Sets an element of w to -inf with probability self.drop_prob.
            Does not drop the diagonal and one of the neighboring elements."""

        dont_drop = torch.eye(w.shape[1], dtype=w.dtype, device=w.device)  # leave the diagonal

        inds = torch.arange(0, w.shape[1], 1, device=w.device)
        shift = torch.randint(low=0, high=2, size=(w.shape[1],), device=w.device)
        shift[0] = 1  # leave right neighbor for y1
        shift[-1] = -1  # leave left neighbor for yM
        shift[shift == 0] = -1  # randomly leave left or right neighbor for y2,...yM-1
        dont_drop[inds, inds+shift] = 1

        prob = torch.ones_like(w) * (1.0 - self.drop_prob)
        prob = torch.clip(prob + dont_drop, 0, 1)

        mask = torch.bernoulli(prob)  # 1 - don't drop, 0 - drop
        mask[mask == 0] = torch.inf
        mask[mask == 1] = 0

        return w - mask

    def update_time_grid(self, t: Tensor) -> None:
        pass

    def _eval_attn_weights(self, x: Tensor) -> Tensor:
        raise NotImplementedError()

    def _eval_output(self, attn_weights: Tensor, x: Tensor) -> Tensor:
        raise NotImplementedError()


class DotProductAttention(AttentionBase):
    def __init__(self, d_model: int, rpe: Union[Module, None] = None, **kwargs):
        super().__init__(d_model, rpe)
        self.W_k = nn.Linear(self.d_model, self.d_model, bias=False)
        self.W_v = nn.Linear(self.d_model, self.d_model, bias=False)
        self.W_q = nn.Linear(self.d_model, self.d_model, bias=False)
        self.W_out = nn.Linear(self.d_model, self.d_model, bias=False)

    def _eval_attn_weights(self, x: Tensor) -> Tensor:
        Q, K = self.W_q(x), self.W_k(x)
        unnorm_attn_weights = torch.bmm(Q, torch.transpose(K, 1, 2)) / self.d_model**0.5
        attn_weights = nn.Softmax(-1)(unnorm_attn_weights)
        return attn_weights

    def _eval_output(self, attn_weights: Tensor, x: Tensor) -> Tensor:
        V = self.W_v(x)
        if self.rpe is None:
            output = torch.bmm(attn_weights, V)
        else:
            output = torch.bmm(attn_weights, V) + (attn_weights.unsqueeze(-1) * self.rpe()).sum(2)
        return self.W_out(output)


class TemporalAttention(AttentionBase):
    def __init__(
        self,
        d_model: int,
        t: Tensor,
        eps: float,
        delta_r: float,
        p: float,
        rpe: Union[Module, None] = None,
        drop_prob: float = 0.0,
        **kwargs
    ) -> None:

        super().__init__(d_model, rpe, drop_prob)
        self.eps = eps
        self.delta_r = delta_r
        self.p = p if p != -1 else torch.inf

        self.W_v = nn.Linear(self.d_model, self.d_model, bias=False)
        self.W_out = nn.Linear(self.d_model, self.d_model, bias=False)

        self.attn_weights: Tensor
        self.update_time_grid(t)

    def _eval_attn_weights(self, x: Tensor) -> Tensor:
        if self.training:
            attn_weights = nn.Softmax(-1)(self.drop(self.unnorm_temporal_attn_weights))
        else:
            attn_weights = nn.Softmax(-1)(self.unnorm_temporal_attn_weights)
        return attn_weights

    def _eval_output(self, attn_weights: Tensor, x: Tensor) -> Tensor:
        assert x.shape[0:2] == attn_weights.shape[0:2], (
            "Batch size and number of time points in `x` and `attn_weights` must be the same. "
            f"Currently {x.shape=} and {attn_weights.shape=}."
        )
        V = self.W_v(x)
        if self.rpe is None:
            output = torch.bmm(attn_weights, V)
        else:
            output = torch.bmm(attn_weights, V) + (attn_weights.unsqueeze(-1) * self.rpe()).sum(2)
        return self.W_out(output)

    @torch.no_grad()
    def update_time_grid(self, t: Tensor) -> None:
        dt = torch.cdist(t, t, p=1).float()
        self.unnorm_temporal_attn_weights = np.log(self.eps) * torch.pow(dt/self.delta_r, self.p)


class TemporalDotProductAttention(AttentionBase):
    def __init__(
        self,
        d_model: int,
        t: Tensor,
        eps: float,
        delta_r: float,
        p: float,
        rpe: Union[Module, None] = None,
        drop_prob: float = 0.0,
        **kwargs
    ) -> None:

        super().__init__(d_model, rpe, drop_prob)
        self.eps = eps
        self.delta_r = delta_r
        self.p = p if p != -1 else torch.inf

        self.W_k = nn.Linear(self.d_model, self.d_model, bias=False)
        self.W_v = nn.Linear(self.d_model, self.d_model, bias=False)
        self.W_q = nn.Linear(self.d_model, self.d_model, bias=False)
        self.W_out = nn.Linear(self.d_model, self.d_model, bias=False)

        self.unnorm_temporal_attn_weights: Tensor
        self.update_time_grid(t)

    def _eval_attn_weights(self, x: Tensor) -> Tensor:
        Q, K = self.W_q(x), self.W_k(x)
        unnorm_dotprod_attn_weights = torch.bmm(Q, torch.transpose(K, 1, 2)) / self.d_model**0.5
        if self.training:
            attn_weights = nn.Softmax(-1)(self.drop(unnorm_dotprod_attn_weights + self.unnorm_temporal_attn_weights))
        else:
            attn_weights = nn.Softmax(-1)(unnorm_dotprod_attn_weights + self.unnorm_temporal_attn_weights)
        return attn_weights

    def _eval_output(self, attn_weights: Tensor, x: Tensor) -> Tensor:
        assert x.shape[0:2] == attn_weights.shape[0:2], (
            "Batch size and number of time points in `x` and `attn_weights` must be the same. "
            f"Currently {x.shape=} and {attn_weights.shape=}."
        )
        V = self.W_v(x)
        if self.rpe is None:
            output = torch.bmm(attn_weights, V)
        else:
            output = torch.bmm(attn_weights, V) + (attn_weights.unsqueeze(-1) * self.rpe()).sum(2)
        return self.W_out(output)

    @torch.no_grad()
    def update_time_grid(self, t: Tensor) -> None:
        dt = torch.cdist(t, t, p=1).float()
        self.unnorm_temporal_attn_weights = np.log(self.eps) * torch.pow(dt/self.delta_r, self.p)


class TemporalDotProductAttentionBaseline(TemporalDotProductAttention):
    def __init__(
        self,
        d_model: int,
        t: Tensor,
        eps: float,
        delta_r: float,
        p: float,
        n: int,
        rpe: Union[Module, None] = None,
        drop_prob: float = 0.0,
        **kwargs
    ) -> None:
        self.n = n
        super().__init__(d_model, t, eps, delta_r, p, rpe, drop_prob, **kwargs)

    @torch.no_grad()
    def update_time_grid(self, t: Tensor) -> None:
        super().update_time_grid(t)
        self.unnorm_temporal_attn_weights += self._create_mask()

    def _create_mask(self) -> Tensor:
        M = self.unnorm_temporal_attn_weights.shape[1]
        device = self.unnorm_temporal_attn_weights.device

        ones = torch.ones((M, M), device=device).triu(self.n+1)
        mask = ones + ones.T
        mask[mask == 1] = -torch.inf

        return mask.unsqueeze(0)
