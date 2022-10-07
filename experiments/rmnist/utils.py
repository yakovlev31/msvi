import os
from collections import deque
import numpy as np
import torch
import msvi.posterior
from einops import rearrange


ndarray = np.ndarray
Tensor = torch.Tensor


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)


def save_model(model, path, name):
    if not os.path.isdir(path):
        os.makedirs(path)
    torch.save(model.state_dict(), path+name+".pt")


def load_model(model, path, name, device):
    model.load_state_dict(torch.load(path+name+".pt", map_location=device), strict=False)


def get_inference_data(t: Tensor, y: Tensor, delta_inf: float) -> tuple[list[Tensor], list[Tensor]]:
    t_inf, y_inf = [], []
    for i in range(t.shape[0]):
        inf_inds = torch.argwhere(t[[i]] <= delta_inf)[:, 1]
        t_inf.append(t[[i]][:, inf_inds, :])
        y_inf.append(y[[i]][:, inf_inds, :, :])
    return t_inf, y_inf


def get_x0(elbo, t: list[Tensor], y: list[Tensor]) -> Tensor:
    x0 = []
    for ti, yi in zip(t, y):
        elbo.q.rec_net.update_time_grids(ti)
        gamma, tau = elbo.q.rec_net(yi)
        x0.append(gamma[:, [0], :] + tau[:, [0], :] * torch.randn_like(tau[:, [0], :]))
    return torch.cat(x0)


def _pred_full_traj(elbo, t: Tensor, x0: Tensor) -> Tensor:
    elbo.p.set_theta(elbo.q.sample_theta())
    S, M, K = x0.shape[0], t.shape[1], x0.shape[2]

    x = torch.zeros((S, M, K), dtype=x0.dtype, device=x0.device)
    x[:, [0], :] = x0

    for i in range(1, M):
        x[:, [i], :] = elbo.p.F(x[:, [i-1], :], t=msvi.posterior.extract_time_grids(t[:, i-1:i+1, :], n_blocks=1))

    return elbo.p._sample_lik(x)


def pred_full_traj(param, elbo, t: Tensor, y: Tensor) -> Tensor:
    t_inf, y_inf = get_inference_data(t, y, param.delta_inf)
    x0 = get_x0(elbo, t_inf, y_inf)
    y_full_traj = _pred_full_traj(elbo, t, x0)
    return y_full_traj


class BatchMovingAverage():
    """Computes moving average over the last `k` mini-batches
    and stores the smallest recorded moving average in `min_avg`."""
    def __init__(self, k: int) -> None:
        self.values = deque([], maxlen=k)
        self.min_avg = np.inf

    def add_value(self, value: float) -> None:
        self.values.append(value)

    def get_average(self) -> float:
        if len(self.values) == 0:
            avg = np.nan
        else:
            avg = sum(self.values) / len(self.values)

        if avg < self.min_avg:
            self.min_avg = avg

        return avg

    def get_min_average(self):
        return self.min_avg


def kl_norm_norm(mu0: Tensor, mu1: Tensor, sig0: Tensor, sig1: Tensor) -> Tensor:
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


def create_mask(x: Tensor) -> Tensor:
    """Masks the 'velocity' part of the latent space since we want to use
    only the 'position' to reconstruct the observsations."""
    K = x.shape[2]
    mask = torch.ones_like(x)
    mask[:, :, K//2:] = 0.0
    return mask


def param_norm(module):
    total_norm = 0.0
    for p in module.parameters():
        if p.requires_grad:
            total_norm += p.data.norm(2).item()
    return total_norm


def grad_norm(module):
    total_norm = 0.0
    for p in module.parameters():
        if p.requires_grad:
            total_norm += p.grad.data.norm(2).item()
    return total_norm


def split_trajectories(t, y, new_traj_len, batch_size):
    s, m, n, d = y.shape
    t_new = torch.empty((s, m-new_traj_len+1, new_traj_len, 1), dtype=t.dtype, device=t.device)
    y_new = torch.empty((s, m-new_traj_len+1, new_traj_len, n, d), dtype=y.dtype, device=y.device)

    for i in range(m - new_traj_len + 1):
        t_new[:, i] = t[:, i:i+new_traj_len]
        y_new[:, i] = y[:, i:i+new_traj_len]

    t_new = rearrange(t_new, "a b c () -> (a b) c ()")
    t_new -= torch.min(t_new, dim=1, keepdim=True)[0]
    y_new = rearrange(y_new, "a b c n d -> (a b) c n d")

    inds = np.random.choice(t_new.shape[0], size=batch_size, replace=False)
    t_new = t_new[inds]
    y_new = y_new[inds]

    return t_new, y_new
