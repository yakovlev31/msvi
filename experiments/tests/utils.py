import numpy as np
import scipy.integrate

import torch
import torch.nn as nn

from torch.nn.parameter import Parameter
from torch.utils.data import DataLoader

from einops import rearrange
from einops.layers.torch import Rearrange

import msvi.decoder
import msvi.trans_func
import msvi.rec_net
import msvi.model
import msvi.posterior
import msvi.elbo

import msvi.utils.utils

from msvi.dataset import TrajectoryDataset


# Use Lotka-Volterra for sanity check.


ndarray = np.ndarray


LV_PARAM = [2.0/3, 4.0/3, 1.0, 1.0]  # parameters of the system
LV_IC = np.array(
    [
        [0.9, 1.8],
        [1.9, 0.9],
        [0.45, 0.9]
    ]
)  # initial conditions


def generate_irregular_time_grid(T, intensity, min_dist):
    """Generates irregular time grid on the interval [0, T].

    Args:
        T (float): Terminal time.
        intensity (float): Intensity of the observations (per second).
        min_dist (float): Smallest distance between time points.

    Returns:
        t (ndarray): 1D array with time points.
    """

    t = [0.0]
    while t[-1] < T:
        t.append(t[-1] + np.random.exponential(1.0/intensity))
    t.pop(-1)
    t[-1] = T

    leave_mask = [True] * len(t)
    for i in range(0, len(t)):
        if leave_mask[i] is True:
            for j in range(i+1, len(t)):
                dist = t[j] - t[i]
                if dist < min_dist:
                    leave_mask[j] = False

    return np.array(t)[leave_mask]


def lv_dynamics(t, x):
    alpha, beta, gamma, delta = LV_PARAM
    dzdt = np.array(
        [
            alpha * x[0] - beta * x[0] * x[1],
            delta * x[0] * x[1] - gamma * x[1],
        ]
    )
    return dzdt


def generate_data(T: float, M: int, sigY: float, seed: int) -> tuple[ndarray, ...]:
    np.random.seed(seed)

    t = np.empty(len(LV_IC), dtype=object)
    x = np.empty(len(LV_IC), dtype=object)
    y = np.empty(len(LV_IC), dtype=object)

    for i in range(len(LV_IC)):

        # ti = np.linspace(0, LV_T, LV_M)
        ti = generate_irregular_time_grid(T, M/T, min_dist=0.02)

        xi = scipy.integrate.solve_ivp(lv_dynamics, ti[[0, -1]], LV_IC[i], method="RK45", rtol=1e-5, atol=1e-5, t_eval=ti).y.T

        t[i] = rearrange(ti, "m -> m ()")
        x[i] = rearrange(xi, "m d -> m () d")
        y[i] = x[i] + sigY * np.random.randn(*x[i].shape)

    return t, x, y


def create_datasets(param) -> TrajectoryDataset:
    t, _, y = generate_data(param.T, param.M, param.sigY, param.seed)
    t = [torch.tensor(ti, dtype=torch.float64) for ti in t]
    y = [torch.tensor(yi, dtype=torch.float32) for yi in y]
    train_dataset = TrajectoryDataset(t, y, max_len=param.max_len)
    return train_dataset


def create_dataloaders(dataset: TrajectoryDataset, param) -> DataLoader:
    dataloader = DataLoader(dataset, batch_size=param.batch_size, shuffle=True)
    return dataloader


def get_model_components(param, construct_h: bool):
    g = Decoder(param.sigY)

    F = msvi.trans_func.ODETransitionFunction(
        f=nn.Sequential(TrueDynamicsFunction()),
        layers_to_count=[TrueDynamicsFunction],
        solver_kwargs=param.solver_kwargs
    )

    if construct_h is True:
        phi_enc = nn.Sequential(Rearrange("s m () d -> s m d"), nn.Linear(2, param.m_h*param.K))
        phi_agg = msvi.utils.utils.create_agg_net(param, "static")
        phi_gamma = nn.Linear(param.m_h*param.K, 2)
        phi_tau = nn.Linear(param.m_h*param.K, 2)
        h = msvi.rec_net.RecognitionNet(phi_enc, phi_agg, phi_gamma, phi_tau, 0)
    else:
        h = None

    return g, F, h


def create_vss_elbo(g, F, param, S):
    prior_param_dict = nn.ParameterDict({
        "mu0": Parameter(0.0 * torch.ones([2]), False),
        "sig0": Parameter(1.0 * torch.ones([2]), False),
        "sigXi": Parameter(0.001 * torch.ones([1]), False),
        "mu_theta": Parameter(0.0 * torch.ones([1]), False),
        "sig_theta": Parameter(1.0 * torch.ones([1]), False),
    })
    posterior_param_dict = nn.ParameterDict({
        "mu_theta_g": Parameter(0.0 * torch.ones(g.param_count())),
        "log_sig_theta_g": Parameter(-7.0 * torch.ones(g.param_count())),
        "mu_theta_F": Parameter(0.0 * torch.ones(F.param_count())),
        "log_sig_theta_F": Parameter(-7.0 * torch.ones(F.param_count())),
        "gamma": Parameter(0.0 * torch.ones([S, 1, 2])),
        "log_tau": Parameter(-7.0 * torch.ones([S, param.max_len-1, 2])),
    })
    p = msvi.model.ModelNormal(prior_param_dict, g, F)
    q = msvi.posterior.SingleShootingPosterior(posterior_param_dict, F)
    elbo = msvi.elbo.SingleShootingELBO(p, q)
    elbo.p.set_theta(elbo.q.sample_theta())
    return elbo


def create_vms_elbo(g, F, param, S):
    prior_param_dict = nn.ParameterDict({
        "mu0": Parameter(0.0 * torch.ones([2]), False),
        "sig0": Parameter(1.0 * torch.ones([2]), False),
        "sigXi": Parameter(0.001 * torch.ones([1]), False),
        "mu_theta": Parameter(0.0 * torch.ones([1]), False),
        "sig_theta": Parameter(1.0 * torch.ones([1]), False),
    })
    posterior_param_dict = nn.ParameterDict({
        "mu_theta_g": Parameter(0.0 * torch.ones(g.param_count())),
        "log_sig_theta_g": Parameter(-7.0 * torch.ones(g.param_count())),
        "mu_theta_F": Parameter(0.0 * torch.ones(F.param_count())),
        "log_sig_theta_F": Parameter(-7.0 * torch.ones(F.param_count())),
        "gamma": Parameter(0.0 * torch.ones([S, param.max_len-1, 2])),
        "log_tau": Parameter(-7.0 * torch.ones([S, param.max_len-1, 2])),
    })
    p = msvi.model.ModelNormal(prior_param_dict, g, F)
    q = msvi.posterior.MultipleShootingPosterior(posterior_param_dict, F)
    elbo = msvi.elbo.MultipleShootingELBO(p, q)
    elbo.p.set_theta(elbo.q.sample_theta())
    return elbo


def create_avms_elbo(g, F, h, param):
    prior_param_dict = nn.ParameterDict({
        "mu0": Parameter(0.0 * torch.ones([2]), False),
        "sig0": Parameter(1.0 * torch.ones([2]), False),
        "sigXi": Parameter(0.001 * torch.ones([1]), False),
        "mu_theta": Parameter(0.0 * torch.ones([1]), False),
        "sig_theta": Parameter(1.0 * torch.ones([1]), False),
    })
    posterior_param_dict = nn.ParameterDict({
        "mu_theta_g": Parameter(0.0 * torch.ones(g.param_count())),
        "log_sig_theta_g": Parameter(-7.0 * torch.ones(g.param_count())),
        "mu_theta_F": Parameter(0.0 * torch.ones(F.param_count())),
        "log_sig_theta_F": Parameter(-7.0 * torch.ones(F.param_count())),
    })
    p = msvi.model.ModelNormal(prior_param_dict, g, F)
    q = msvi.posterior.AmortizedMultipleShootingPosterior(posterior_param_dict, F, h)
    elbo = msvi.elbo.AmortizedMultipleShootingELBO(p, q)
    elbo.p.set_theta(elbo.q.sample_theta())
    return elbo


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)


class TrueDynamicsFunction(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = Parameter(torch.zeros(4))  # alpha, beta, gamma, delta
        self.bias = Parameter(torch.zeros(1))  # dummy parameter required for compatibility with msvi.trans_func

    def forward(self, x):
        alpha, beta, gamma, delta = self.weight
        x1, x2 = x[..., [0]], x[..., [1]]
        dxdt = torch.zeros_like(x)
        dxdt[..., [0]] = alpha * x1 - beta * x1 * x2
        dxdt[..., [1]] = delta * x1 * x2 - gamma * x2
        return dxdt


class Decoder(msvi.decoder.IDecoder):
    def __init__(self, sigY: float) -> None:
        super().__init__()
        self.sigY = sigY

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        S, M, D = x.shape
        p = torch.empty((S, M, 1, D, 2), device=x.device)
        p[:, :, 0, :, 0] = x
        p[:, :, 0, :, 1] = self.sigY
        return p

    def set_param(self, param: torch.Tensor) -> None:
        return None

    def param_count(self) -> int:
        return 0
