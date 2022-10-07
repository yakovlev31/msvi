from typing import Union

from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from torch.distributions.normal import Normal
from torch.distributions.bernoulli import Bernoulli
from torch.distributions.continuous_bernoulli import ContinuousBernoulli

from einops import reduce

from msvi.decoder import IDecoder
from msvi.trans_func import ITransitionFunction
from msvi.posterior import extract_time_grids


Tensor = torch.Tensor
ParameterDict = nn.ParameterDict


class IModel(ABC, nn.Module):
    @property
    @abstractmethod
    def g(self) -> IDecoder:
        """Returns the decoder."""
        pass

    @property
    @abstractmethod
    def F(self) -> ITransitionFunction:
        """Returns the transition function."""
        pass

    @property
    @abstractmethod
    def prior_param(self) -> ParameterDict:
        """Returns parameters of prior distributions."""
        pass

    @abstractmethod
    def sample(self, t: Tensor, x0: Tensor) -> Tensor:
        """Samples a trajectory from the model. If x0=None, uses the prior to
            sample the initial condition.

        Args:
            t: Time points at which to evaluate the trajectory. Has shape (M, ).
            x0: Initial condition. Has shape (K, ).

        Returns:
            Trajectory sampled from the model. Has shape (1, M, N, D).
        """
        pass

    @abstractmethod
    def loglik(self, y: Tensor, x: Tensor) -> Tensor:
        """Evaluates log likelihood p(y|x) for each snapshot.

        Args:
            y: Observations. Has shape (S, M, N, D).
            x: Latent states. Has shape (S, M, K).

        Returns:
            Log likelihood for each snapshot. Has shape (S, M, 1).
        """
        pass

    @abstractmethod
    def set_theta(self, theta: dict[str, Tensor]) -> None:
        """Sets parameters of g and F to theta["theta_g"] and theta["theta_F"] respectively.

        Args:
            theta: Dictionary with new parameter values. Must contain keys
                theta_g and theta_F.
        """
        pass


class ModelBase(IModel):
    def __init__(
        self,
        prior_param_dict: ParameterDict,
        g: IDecoder,
        F: ITransitionFunction,
    ) -> None:
        super().__init__()
        self._check_param_shapes(prior_param_dict)
        self._prior_param = prior_param_dict
        self._g = g
        self._F = F

    @property
    def g(self) -> IDecoder:
        return self._g

    @property
    def F(self) -> ITransitionFunction:
        return self._F

    @property
    def prior_param(self) -> ParameterDict:
        return self._prior_param

    def sample(self, t: Tensor, x0: Tensor) -> Tensor:
        x = self._sample_x(t, x0)
        y = self._sample_lik(x)
        return y

    def loglik(self, y: Tensor, x: Tensor) -> Tensor:
        return self._eval_loglik(y, x)

    def set_theta(self, theta: dict[str, Tensor]) -> None:
        self.g.set_param(theta["theta_g"])
        self.F.set_param(theta["theta_F"])

    def _sample_x(self, t: Tensor, x0: Union[None, Tensor] = None) -> Tensor:
        if x0 is None:
            x0 = self._sample_ic()
        x = self._sample_traj(t, x0)
        return x

    def _sample_ic(self):
        mu0, sig0 = self.prior_param["mu0"], self.prior_param["sig0"]
        x0 = mu0 + sig0 * torch.randn_like(sig0)
        return x0

    def _sample_traj(self, t, x0):
        x = torch.empty((1, t.shape[0], x0.shape[0]), device=x0.device)
        x[0, 0, :] = x0
        s_curr = x0
        for i in range(1, t.shape[0]):
            x[:, [i], :] = self.F(s_curr, t=extract_time_grids(t[:, i-1:i+1, :], n_blocks=1))
            eps = self.prior_param["sigXi"] * torch.randn_like(x[:, [i], :])
            s_curr = x[:, [i], :] + eps
        return x

    def _check_param_shapes(self, d: ParameterDict) -> None:
        scalar_param_names = ["sigXi", "mu_theta", "sig_theta"]
        for param_name in scalar_param_names:
            assert d[param_name].shape == torch.Size([1]), f"{param_name} must have shape (1, ) but has {d[param_name].shape}."
        assert len(d["mu0"].shape) == 1, f"mu0 must have shape (K, ) but has {d['mu0'].shape}."
        assert len(d["sig0"].shape) == 1, f"sig0 must have shape (K, ) but has {d['sig0'].shape}."

    def _sample_lik(self, x: Tensor) -> Tensor:
        raise NotImplementedError()

    def _eval_loglik(self, y: Tensor, x: Tensor) -> Tensor:
        raise NotImplementedError()


class ModelNormal(ModelBase):
    def _sample_lik(self, x: Tensor) -> Tensor:
        param = self.g(x)
        mu, sig = param[..., 0], param[..., 1]
        y = Normal(mu, sig).rsample()
        return y

    def _eval_loglik(self, y: Tensor, x: Tensor) -> Tensor:
        param = self.g(x)
        mu, sig = param[..., 0], param[..., 1]
        loglik = Normal(mu, sig).log_prob(y)
        loglik = reduce(loglik, "s m n d -> s m ()", "sum")
        return loglik


class ModelNormalSecondOrder(ModelNormal):
    def _sample_lik(self, x: Tensor) -> Tensor:
        mask = self.create_mask(x)
        return super()._sample_lik(x * mask)

    def _eval_loglik(self, y: Tensor, x: Tensor) -> Tensor:
        mask = self.create_mask(x)
        return super()._eval_loglik(y, x * mask)

    def create_mask(self, x: Tensor) -> Tensor:
        """Masks the 'velocity' part of the latent space since we want to use
        only the 'position' to reconstruct the observsations."""
        K = x.shape[2]
        mask = torch.ones_like(x)
        mask[:, :, K//2:] = 0.0
        return mask


class ModelBernoulli(ModelBase):
    def _sample_lik(self, x: Tensor) -> Tensor:
        p = self.g(x)[..., 0]
        y = Bernoulli(p).rsample()
        return y

    def _eval_loglik(self, y: Tensor, x: Tensor) -> Tensor:
        p = self.g(x)[..., 0]
        loglik = Bernoulli(p).log_prob(y)
        loglik = reduce(loglik, "s m n d -> s m ()", "sum")
        return loglik


class ModelContinuousBernoulli(ModelBase):
    def _sample_lik(self, x: Tensor) -> Tensor:
        p = self.g(x)[..., 0]
        y = ContinuousBernoulli(p).rsample()
        return y

    def _eval_loglik(self, y: Tensor, x: Tensor) -> Tensor:
        p = self.g(x)[..., 0]
        loglik = ContinuousBernoulli(p).log_prob(y)
        loglik = reduce(loglik, "s m n d -> s m ()", "sum")
        return loglik
