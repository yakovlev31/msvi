import torch
import torch.nn as nn

from einops import rearrange


Tensor = torch.Tensor
Module = nn.Module


class RecognitionNet(Module):
    def __init__(
        self,
        phi_enc: Module,
        phi_agg: Module,
        phi_gamma: Module,
        phi_tau: Module,
        tau_min: float,
    ) -> None:
        """This class is used to convert observations to variational parameters.

        There are four main components:
            phi_enc: a point-wise encoder which maps y:(S, M, N, D) to a:(S, M, K').
            phi_agg: a sequence to sequence function which maps a:(S, M, K') to b:(S, M, K').
            phi_gamma/phi_tau: a point-wise function which maps b:(S, M, K') to gamma/tau:(S, M, K).

        First, observations `y` are converted to a lower-dimensional form `a` by the encoder `phi_enc`.
        Then, sequence `a` is aggregated into another sequence `b` by `phi_agg`.
        Finally, variational parameters are extracted from `b` by `phi_gamma` and `phi_tau`.
        """

        super().__init__()
        self.phi_enc = phi_enc
        self.phi_agg = phi_agg
        self.phi_gamma = phi_gamma
        self.phi_tau = phi_tau

        self.tau_min = tau_min

    def forward(self, y: Tensor) -> tuple[Tensor, Tensor]:
        """Converts observations to variational parameters.

            Args:
                y: Observations. Has shape (S, M, N, D).

            Returns:
                gamma: Variational parameters. Has shape (S, M, K).
                tau: Variational parameters. Has shape (S, M, K).
        """
        a = self.phi_enc(y)
        b = self.phi_agg(a)

        gamma = self.phi_gamma(b)
        tau = torch.exp(self.phi_tau(b)) + self.tau_min

        return gamma, tau

    def apply_batch_norm(self, gamma, bn):
        S, M, _ = gamma.shape
        gamma = rearrange(gamma, "s m k -> (s m) k")
        gamma = bn(gamma)
        gamma = rearrange(gamma, "(s m) k -> s m k", s=S, m=M)
        return gamma

    def update_time_grids(self, t: Tensor) -> None:
        """Updates all parts of aggregation net that depend on time grids."""
        for module in self.phi_agg.modules():
            if not hasattr(module, "update_time_grid"):
                continue
            if callable(getattr(module, "update_time_grid")):
                module.update_time_grid(t)  # type: ignore


class RecognitionNetSecondOrder(RecognitionNet):
    """Same as RecognitionNet but splits variational parameters into two groups."""
    def __init__(
        self,
        phi_enc: Module,
        phi_agg: Module,
        phi_gamma: Module,
        phi_tau: Module,
        phi_agg_dyn: Module,
        phi_gamma_dyn: Module,
        phi_tau_dyn: Module,
        tau_min: float,
    ) -> None:

        super().__init__(phi_enc, phi_agg, phi_gamma, phi_tau, tau_min)

        self.phi_agg_dyn = phi_agg_dyn
        self.phi_gamma_dyn = phi_gamma_dyn
        self.phi_tau_dyn = phi_tau_dyn

    def forward(self, y: Tensor) -> tuple[Tensor, Tensor]:
        a = self.phi_enc(y)

        b_stat = self.phi_agg(a)
        b_dyn = self.phi_agg_dyn(a)

        gamma_stat = self.phi_gamma(b_stat)
        tau_stat = torch.exp(self.phi_tau(b_stat)) + self.tau_min

        gamma_dyn = self.phi_gamma_dyn(b_dyn)
        tau_dyn = torch.exp(self.phi_tau_dyn(b_dyn))

        gamma = torch.cat((gamma_stat, gamma_dyn), dim=2)
        tau = torch.cat((tau_stat, tau_dyn), dim=2)

        return gamma, tau

    def update_time_grids(self, t: Tensor) -> None:
        for agg_net in [self.phi_agg, self.phi_agg_dyn]:
            for module in agg_net.modules():
                if not hasattr(module, "update_time_grid"):
                    continue
                if callable(getattr(module, "update_time_grid")):
                    module.update_time_grid(t)  # type: ignore


def set_module_requires_grad(m: Module, value: bool):
    for p in m.parameters():
        p.requires_grad = value
