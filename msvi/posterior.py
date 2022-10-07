from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from msvi.trans_func import ITransitionFunction
from msvi.rec_net import RecognitionNet


Tensor = torch.Tensor
Module = nn.Module
ParameterDict = nn.ParameterDict


class IVariationalPosterior(ABC, Module):
    @property
    @abstractmethod
    def posterior_param(self) -> nn.ParameterDict:
        """Returns parameters of the posterior distribution."""
        pass

    @abstractmethod
    def sample_s(
        self,
        t: Tensor,
        y: Tensor,
        batch_ids: Tensor,
        block_size: int,
    ) -> tuple[Tensor, Tensor]:
        """Samples shooting variables (s_1, ..., s_B) from the posterior q(s|y).
        Also returns states (x_1, ..., x_M).

        Args:
            t: Time points at which to evaluate the latent states. Has shape (S, M, 1).
            y: Observations corresponding to the latent states. Used only for
                amortized variational inference. Has shape (S, M, N, D).
            batch_ids: Indices of the trajectories for which to sample the shooting variables.
                Has shape (S, ).
            block_size: Size of the blocks.

        Returns:
            A sample of the shooting variables with shape (S, B, K)
                and the corresponding latent states with shape (S, M, K).
        """
        pass

    @abstractmethod
    def sample_theta(self) -> dict[str, Tensor]:
        """Samples parameters of g and F from the posterior.

        Returns:
            Dictionary with a sample of the parameters.
        """
        pass


class VariationalPosteriorBase(IVariationalPosterior):
    def __init__(self, posterior_param_dict: ParameterDict):
        super().__init__()
        self._check_param_shapes(posterior_param_dict)
        self._posterior_param = posterior_param_dict

    @property
    def posterior_param(self):
        return self._posterior_param

    def sample_theta(self):
        mu_g, sig_g = self.posterior_param["mu_theta_g"], torch.exp(self.posterior_param["log_sig_theta_g"])
        mu_F, sig_F = self.posterior_param["mu_theta_F"], torch.exp(self.posterior_param["log_sig_theta_F"])
        theta = {
            "theta_g": mu_g + sig_g * torch.randn_like(sig_g),
            "theta_F": mu_F + sig_F * torch.randn_like(sig_F),
        }
        return theta

    def _check_param_shapes(self, p: ParameterDict) -> None:
        raise NotImplementedError()

    def sample_s(self, t: Tensor, y: Tensor, batch_ids: Tensor, block_size: int) -> tuple[Tensor, Tensor]:
        raise NotImplementedError()


class SingleShootingPosterior(VariationalPosteriorBase):
    def __init__(
        self,
        posterior_param_dict: ParameterDict,
        F: ITransitionFunction,
    ) -> None:

        super().__init__(posterior_param_dict)
        self.F = F

    def sample_s(
        self,
        t: Tensor,
        y: Tensor,
        batch_ids: Tensor,
        block_size: int,
    ) -> tuple[Tensor, Tensor]:

        gamma_0 = self.posterior_param["gamma"][batch_ids]
        tau = torch.exp(self.posterior_param["log_tau"][batch_ids])

        S, M, K = batch_ids.shape[0], t.shape[1], gamma_0.shape[2]
        s = torch.zeros((S, M-1, K), device=tau.device)
        x = torch.zeros((S, M, K), device=tau.device)

        s[:, [0], :] = gamma_0 + tau[:, [0], :] * torch.randn((S, 1, K), device=tau.device)
        x[:, [0], :] = s[:, [0], :]

        for i in range(1, M):
            x_i = self.F(s[:, [i-1], :], t=extract_time_grids(t[:, i-1:i+1, :], n_blocks=1))
            x[:, [i], :] = x_i
            if i < (M - 1):
                s[:, [i], :] = x_i + tau[:, [i], :] * torch.randn((S, 1, K), device=tau.device)

        return s, x

    def _check_param_shapes(self, p: dict[str, Tensor]) -> None:
        model_parameter_names = ["mu_theta_g", "mu_theta_F", "log_sig_theta_g", "log_sig_theta_F"]
        for param_name in model_parameter_names:
            assert len(p[param_name].shape) == 1, f"{param_name} must have shape (num_parameters, ) but has {p[param_name].shape}."
        assert len(p["gamma"].shape) == 3 and p["gamma"].shape[1] == 1, f"gamma must have shape (S, 1, K) but has {p['gamma'].shape}."
        assert len(p["log_tau"].shape) == 3, f"log_tau must have shape (S, M-1, K) but has {p['log_tau'].shape}."


class MultipleShootingPosterior(VariationalPosteriorBase):
    def __init__(
        self,
        posterior_param_dict: ParameterDict,
        F: ITransitionFunction
    ) -> None:

        super().__init__(posterior_param_dict)
        self.F = F

    def sample_s(
        self,
        t: Tensor,
        y: Tensor,
        batch_ids: Tensor,
        block_size: int,
    ) -> tuple[Tensor, Tensor]:

        gamma = self.posterior_param["gamma"][batch_ids, ::block_size, :]
        tau = torch.exp(self.posterior_param["log_tau"][batch_ids, ::block_size, :])
        s = gamma + tau * torch.randn_like(gamma)

        S, M, B, K = batch_ids.shape[0], t.shape[1], gamma.shape[1], gamma.shape[2]
        x = torch.zeros((S, M, K), device=tau.device)

        x[:, [0], :] = s[:, [0], :]
        x[:, 1:, :] = self.F(s, t=extract_time_grids(t, n_blocks=B))

        return s, x

    def _check_param_shapes(self, p: dict[str, Tensor]) -> None:
        model_parameter_names = ["mu_theta_g", "mu_theta_F", "log_sig_theta_g", "log_sig_theta_F"]
        for param_name in model_parameter_names:
            assert len(p[param_name].shape) == 1, f"{param_name} must have shape (num_parameters, ) but has {p[param_name].shape}."
        assert len(p["gamma"].shape) == 3, f"gamma must have shape (S, M, K) but has {p['gamma'].shape}."
        assert p["gamma"].shape == p["log_tau"].shape, f"shapes of gamma ({p['gamma'].shape}) and log_tau ({p['log_tau'].shape}) must be the same."


class AmortizedMultipleShootingPosterior(VariationalPosteriorBase):
    def __init__(
        self,
        posterior_param_dict: ParameterDict,
        F: ITransitionFunction,
        rec_net: RecognitionNet,
    ) -> None:

        super().__init__(posterior_param_dict)
        self.F = F
        self.rec_net = rec_net

        self.gamma: Tensor
        self.tau: Tensor

    def sample_s(
        self,
        t: Tensor,
        y: Tensor,
        batch_ids: Tensor,
        block_size: int,
    ) -> tuple[Tensor, Tensor]:

        assert y is not None, "Amortized posterior requires data y."

        gamma, tau = self.rec_net(y)
        self.gamma, self.tau = gamma[:, :-1, :], tau[:, :-1, :]

        gamma = self.gamma[:, ::block_size, :]
        tau = self.tau[:, ::block_size, :]
        s = gamma + tau * torch.randn_like(tau)

        S, M, B, K = batch_ids.shape[0], t.shape[1], gamma.shape[1], gamma.shape[2]
        x = torch.zeros((S, M, K), device=tau.device)

        x[:, [0], :] = s[:, [0], :]
        x[:, 1:, :] = self.F(s, t=extract_time_grids(t, n_blocks=B))

        return s, x

    def _check_param_shapes(self, p: dict[str, Tensor]) -> None:
        model_parameter_names = ["mu_theta_g", "mu_theta_F", "log_sig_theta_g", "log_sig_theta_F"]
        for param_name in model_parameter_names:
            assert len(p[param_name].shape) == 1, f"{param_name} must have shape (num_parameters, ) but has {p[param_name].shape}."


def extract_time_grids(t: Tensor, n_blocks: int) -> Tensor:
    """Extracts overlapping sub-grids from `t` for the given number of blocks.

    Args:
        t: Full time grids. Has shape (S, M, 1).
        n_blocks: Number of blocks.

    Returns:
        sub_t: Overlapping sub-grids. Has shape (S, n_blocks, grid_size).

    Simplified example:
        For t=(t1, t2, t3, t4, t5) and b_blocks=2 returns (t1, t2, t3), (t3, t4, t5).
    """

    S, M = t.shape[0:2]
    assert (M - 1) % n_blocks == 0, "All blocks must be of equal size."

    grid_size = int((M - 1) / n_blocks) + 1
    sub_t = torch.empty((S, n_blocks, grid_size), dtype=t.dtype, device=t.device)

    for b, i in enumerate(range(0, M-grid_size+1, grid_size-1)):
        sub_t[:, [b], :] = torch.transpose(t[:, i:i+grid_size, :], 1, 2)

    return sub_t
