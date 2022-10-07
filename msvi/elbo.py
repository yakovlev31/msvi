from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from msvi.model import IModel
from msvi.posterior import IVariationalPosterior, AmortizedMultipleShootingPosterior

from einops import repeat


Tensor = torch.Tensor


class IELBO(nn.Module, ABC):
    @abstractmethod
    def forward(
        self,
        t: Tensor,
        y: Tensor,
        batch_ids: Tensor,
        block_size: int,
        scaler: float = 1.0,
    ) -> tuple[Tensor, ...]:
        """Evaluates ELBO for the observations y.

        Args:
            t: Time grid for the observations. Has shape (S, M, 1).
            y: A batch of observations. Has shape (S, M, N, D).
            batch_ids: Global indices of trajectories in the batch. Has shape (S, ).
            block_size: Block size.
            scaler: Scaler for KL(q(s_i)||p(s_i|s_i-1)) terms.

        Returns:
            Parts of the ELBO (L1, L2, L3), states (x), and shooting variables (s).
        """
        pass


class ELBOBase(IELBO):
    def __init__(
        self,
        p: IModel,
        q: IVariationalPosterior,
    ) -> None:

        super().__init__()
        self.p = p
        self.q = q

    def forward(
        self,
        t: Tensor,
        y: Tensor,
        batch_ids: Tensor,
        block_size: int,
        scaler: float = 1.0,
    ) -> tuple[Tensor, ...]:

        # Sample approximate posterior.
        self.p.set_theta(self.q.sample_theta())
        s, x = self.q.sample_s(t, y, batch_ids, block_size)

        # Calculate parts of ELBO.
        L1 = self.calc_L1(x, y)
        L2 = self.calc_L2(x, batch_ids, block_size, scaler)
        L3 = self.calc_L3()

        return L1, L2, L3, x, s

    def calc_L1(self, x: Tensor, y: Tensor) -> Tensor:
        return self.p.loglik(y, x).sum()

    def calc_L2(self, x: Tensor, batch_ids: Tensor, block_size: int, scaler: float) -> Tensor:
        raise NotImplementedError()

    def calc_L3(self) -> Tensor:
        n = self.q.posterior_param["mu_theta_g"].numel()
        L3_0 = self.kl_norm_norm(
            self.q.posterior_param["mu_theta_g"],
            self.p.prior_param["mu_theta"].expand(n),
            torch.exp(self.q.posterior_param["log_sig_theta_g"]),
            self.p.prior_param["sig_theta"].expand(n),
        ).sum()

        n = self.q.posterior_param["mu_theta_F"].numel()
        L3_1 = self.kl_norm_norm(
            self.q.posterior_param["mu_theta_F"],
            self.p.prior_param["mu_theta"].expand(n),
            torch.exp(self.q.posterior_param["log_sig_theta_F"]),
            self.p.prior_param["sig_theta"].expand(n),
        ).sum()
        return L3_0 + L3_1

    def kl_norm_norm(self, mu0: Tensor, mu1: Tensor, sig0: Tensor, sig1: Tensor) -> Tensor:
        """Calculates KL divergence between two K-dimensional Normal
            distributions with diagonal covariance matrices.

        Args:
            mu0: Mean of the first distribution. Has shape (*, K).
            mu1: Mean of the second distribution. Has shape (*, K).
            std0: Diagonal of the covatiance matrix of the first distribution. Has shape (*, K).
            std1: Diagonal of the covatiance matrix of the second distribution. Has shape (*, K).

        Returns:
            KL divergence between the distributions. Has shape (*, 1).
        """
        assert mu0.shape == mu1.shape == sig0.shape == sig1.shape, (f"{mu0.shape=} {mu1.shape=} {sig0.shape=} {sig1.shape=}")
        a = (sig0 / sig1).pow(2).sum(-1, keepdim=True)
        b = ((mu1 - mu0).pow(2) / sig1**2).sum(-1, keepdim=True)
        c = 2 * (torch.log(sig1) - torch.log(sig0)).sum(-1, keepdim=True)
        kl = 0.5 * (a + b + c - mu0.shape[-1])
        return kl


class SingleShootingELBO(ELBOBase):
    def calc_L2(self, x: Tensor, batch_ids: Tensor, block_size: int, scaler: float) -> Tensor:
        S, M, K = x.shape

        gamma = self.q.posterior_param["gamma"][batch_ids]
        tau = torch.exp(self.q.posterior_param["log_tau"][batch_ids])

        L2_0 = self.kl_norm_norm(
            gamma[:, 0, :],
            repeat(self.p.prior_param["mu0"], "k -> s k", s=S, k=K),
            tau[:, 0, :],
            repeat(self.p.prior_param["sig0"], "k -> s k", s=S, k=K)
        ).sum()

        L2_1 = self.kl_norm_norm(
            x[:, 1:-1, :],
            x[:, 1:-1, :],
            tau[:, 1:, :],
            repeat(self.p.prior_param["sigXi"], "() -> s m k", s=S, m=M-2, k=K)
        ).sum()

        return L2_0 + scaler * L2_1


class MultipleShootingELBO(ELBOBase):
    def calc_L2(self, x: Tensor, batch_ids: Tensor, block_size: int, scaler: float) -> Tensor:
        gamma = self.q.posterior_param["gamma"][batch_ids, ::block_size, :]
        tau = torch.exp(self.q.posterior_param["log_tau"][batch_ids, ::block_size, :])

        x_sub = x[:, 0:-1:block_size, :]
        S, B, K = x_sub.shape

        L2_0 = self.kl_norm_norm(
            gamma[:, 0, :],
            repeat(self.p.prior_param["mu0"], "k -> s k", s=S, k=K),
            tau[:, 0, :],
            repeat(self.p.prior_param["sig0"], "k -> s k", s=S, k=K)
        ).sum()

        L2_1 = self.kl_norm_norm(
            gamma[:, 1:, :],
            x_sub[:, 1:, :],
            tau[:, 1:, :],
            repeat(self.p.prior_param["sigXi"], "() -> s b k", s=S, b=B-1, k=K)
        ).sum()

        return L2_0 + scaler * L2_1


class AmortizedMultipleShootingELBO(ELBOBase):
    def __init__(self, p: IModel, q: AmortizedMultipleShootingPosterior) -> None:
        super().__init__(p, q)
        self.q = q

    def forward(
        self,
        t: Tensor,
        y: Tensor,
        batch_ids: Tensor,
        block_size: int,
        scaler: float = 1.0,
    ) -> tuple[Tensor, ...]:

        self.q.rec_net.update_time_grids(t)  # update recognition network before sampling s
        return super().forward(t, y, batch_ids, block_size, scaler)

    def calc_L2(self, x: Tensor, batch_ids: Tensor, block_size: int, scaler: float) -> Tensor:
        gamma = self.q.gamma[:, ::block_size, :]
        tau = self.q.tau[:, ::block_size, :]

        x_sub = x[:, 0:-1:block_size, :]
        S, B, K = x_sub.shape

        L2_0 = self.kl_norm_norm(
            gamma[:, 0, :],
            repeat(self.p.prior_param["mu0"], "k -> s k", s=S, k=K),
            tau[:, 0, :],
            repeat(self.p.prior_param["sig0"], "k -> s k", s=S, k=K)
        ).sum()

        L2_1 = self.kl_norm_norm(
            gamma[:, 1:, :],
            x_sub[:, 1:, :],
            tau[:, 1:, :],
            repeat(self.p.prior_param["sigXi"], "() -> s b k", s=S, b=B-1, k=K)
        ).sum()

        return L2_0 + scaler * L2_1
