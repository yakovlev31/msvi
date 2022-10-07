import os
import pickle
import argparse

from typing import Union
from types import SimpleNamespace

import torch
import torch.nn as nn
import torchvision.transforms
from torch.nn.parameter import Parameter
from torch.utils.data import DataLoader

import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from einops import rearrange

from msvi.model import ModelNormal, ModelNormalSecondOrder
from msvi.posterior import AmortizedMultipleShootingPosterior
from msvi.elbo import AmortizedMultipleShootingELBO
from msvi.decoder import NeuralDecoder
from msvi.trans_func import ODETransitionFunction, ODETransitionFunctionSecondOrder
from msvi.rec_net import RecognitionNet, RecognitionNetSecondOrder
from msvi.dataset import TrajectoryDataset

from msvi.utils.utils import create_agg_net, Sine, CNNEncoder, CNNDecoder


plt.style.use("seaborn")  # type: ignore
sns.set_style("whitegrid")


ndarray = np.ndarray
Tensor = torch.Tensor
Sequential = nn.Sequential
DataDict = dict[str, dict[str, list]]
TensorDataDict = dict[str, dict[str, list[Tensor]]]
Module = nn.Module


DATASET_NAME = "PENDULUM"


def create_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    # Data.
    parser.add_argument("--data_folder", type=str, default="./experiments/data/datasets/pendulum/", help="Path to the dataset.")
    parser.add_argument("--N", type=int, default=1024, help="Number of observation points.")
    parser.add_argument("--D", type=int, default=1, help="Dimensionality of observation.")
    parser.add_argument("--max_len", type=int, default=None, help="Truncation length for trajectories.")
    parser.add_argument("--sigY", type=float, default=1e-3, help="Observation noise.")

    # Model (common).
    parser.add_argument("--K", type=int, default=32, help="Latent space dimension.")
    parser.add_argument("--Xi", type=float, default=1e-4, help="Used to set variance for the continuity prior.")
    parser.add_argument("--block_size", type=int, default=1, help="Number of time points in each block.")

    # Model (g).
    parser.add_argument("--g_cnn_channels", type=int, default=8, help="Channels in CNNDecoder.")

    # Model (F).
    parser.add_argument("--m_F", type=int, default=8, help="Dimensionality scaler for F.")
    parser.add_argument("--F_nonlin", type=str, default="relu", help="Nonlinearity for F.")
    parser.add_argument("--dyn_order", type=int, default=2, help="Order of the dynamcis function, must be 1 or 2.")

    # Model (h).
    parser.add_argument("--m_h", type=int, default=4, help="Dimensionality scaler for h.")
    parser.add_argument("--h_enc_cnn_channels", type=int, default=8, help="Channels in CNNEncoder.")
    parser.add_argument("--h_agg_attn", type=str, default="tdp", help="Attention type (dp, t, tdp, tdp_b).")
    parser.add_argument("--h_agg_pos_enc", type=str, default="rpeNN", help="Position encoding type (csc, rpeNN, rpeInterp).")
    parser.add_argument("--h_agg_stat_layers", type=int, default=4, help="Number of TFEncoder layers in static aggregation net.")
    parser.add_argument("--h_agg_dyn_layers", type=int, default=8, help="Number of TFEncoder layers in dynamic aggregation net.")
    parser.add_argument("--h_agg_max_tokens", type=int, default=51, help="Maximum expected number of tokens.")
    parser.add_argument("--h_agg_max_time", type=float, default=3.0, help="Maximum expected observation time.")
    parser.add_argument("--h_agg_delta_r", type=float, default=0.45, help="Attention time span at training time.")
    parser.add_argument("--h_agg_p", type=float, default=-1, help="Exponent for temporal attention (use -1 for p=inf).")
    parser.add_argument("--n", type=int, default=1, help="Number of nearest neighbors used for baseline aggregation net.")
    parser.add_argument("--drop_prob", type=float, default=0.1, help="Attention dropout probability.")  # 0.1
    parser.add_argument("--tau_min", type=float, default=2e-2, help="Lower bound on the variance of q(s_i).")  # 2e-2
    parser.add_argument("--sigT", type=float, default=0.0, help="Scale of the noise added to the time grids for temporal neighborhood adjustment.")  # 0.00025

    # Training/validation/testing.
    parser.add_argument("--scaler", type=float, default=1, help="Scaler for ELBO L2 term.")
    parser.add_argument("--n_iters", type=int, default=300000, help="Number of training iterations.")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate.")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size.")

    parser.add_argument("--solver", type=str, default="dopri5", help="Name of the ODE solver (see torchdiffeq).")
    parser.add_argument("--rtol", type=float, default=1e-5, help="Relative tolerance for ODE solver.")
    parser.add_argument("--atol", type=float, default=1e-5, help="Absolute tolerance for ODE solver.")
    parser.add_argument("--adjoint", type=int, default=0, help="Use adjoint to evaluate gradient flag (0 - no, 1 - yes).")

    parser.add_argument("--device", type=str, default="cuda", help="Device (cpu or cuda)")
    parser.add_argument("--seed", type=int, default=13, help="Random seed.")
    parser.add_argument("--group", default="None", help="Group for wandb.")
    parser.add_argument("--tags", default=["no_tag"], nargs="+", help="Tags for wandb.")
    parser.add_argument("--name", type=str, default="tmp", help="Name of the run.")

    parser.add_argument("--visualize", type=int, default=1, help="Visualize predictions on validation set flag (0 - no, 1 - yes).")
    parser.add_argument("--n_mc_samples", type=int, default=10, help="Number of samples for Monte Carlo integration.")
    parser.add_argument("--delta_inf", type=float, default=0.45, help="Fraction of obsevations used for x0 inference at test time.")

    parser.add_argument("--model_folder", type=str, default="./models/pendulum/", help="Folder for saving/loading models.")

    return parser


def create_datasets(param: SimpleNamespace, device=None) -> tuple[TrajectoryDataset, ...]:
    data = read_data(param.data_folder)
    data = preprocess_data(data)
    data = to_tensors(data, device)
    train_dataset = TrajectoryDataset(data["train"]["t"], data["train"]["y"], param.max_len)
    val_dataset = TrajectoryDataset(data["val"]["t"], data["val"]["y"], param.max_len)
    test_dataset = TrajectoryDataset(data["test"]["t"], data["test"]["y"], param.max_len)
    return train_dataset, val_dataset, test_dataset


def read_data(path: str) -> DataDict:
    """Reads data from folder `path` which contains subfolders train, val and test.
    Each subfolder contains ndarrays with time grids and trajectories stored as
    t.pkl and y.pkl files."""
    data = {}
    data["train"] = read_pickle(["t", "y"], path+"train/")
    data["val"] = read_pickle(["t", "y"], path+"val/")
    data["test"] = read_pickle(["t", "y"], path+"test/")
    return data


def preprocess_data(data: DataDict) -> DataDict:
    data["train"], train_stats = _preprocess_data(data["train"])
    data["val"], _ = _preprocess_data(data["val"], train_stats)
    data["test"], _ = _preprocess_data(data["test"], train_stats)
    return data


def add_noise(data: DataDict, sig: float, seed: int) -> DataDict:
    np.random.seed(seed)
    for i in range(len(data["train"]["y"])):
        data["train"]["y"][i] += np.random.randn(*data["train"]["y"][i].shape) * sig
    for i in range(len(data["val"]["y"])):
        data["val"]["y"][i] += np.random.randn(*data["val"]["y"][i].shape) * sig
    for i in range(len(data["test"]["y"])):
        data["test"]["y"][i] += np.random.randn(*data["test"]["y"][i].shape) * sig
    return data


def to_tensors(data: DataDict, device=None) -> TensorDataDict:
    tensor_data = {}
    tensor_data["train"] = {
        "t": [torch.tensor(ti, dtype=torch.float64).to(device) for ti in data["train"]["t"]],
        "y": [torch.tensor(yi, dtype=torch.float32).to(device) for yi in data["train"]["y"]],
    }
    tensor_data["val"] = {
        "t": [torch.tensor(ti, dtype=torch.float64).to(device) for ti in data["val"]["t"]],
        "y": [torch.tensor(yi, dtype=torch.float32).to(device) for yi in data["val"]["y"]],
    }
    tensor_data["test"] = {
        "t": [torch.tensor(ti, dtype=torch.float64).to(device) for ti in data["test"]["t"]],
        "y": [torch.tensor(yi, dtype=torch.float32).to(device) for yi in data["test"]["y"]],
    }
    return tensor_data


def create_dataloaders(
    param: SimpleNamespace,
    train_dataset: TrajectoryDataset,
    val_dataset: TrajectoryDataset,
    test_dataset: TrajectoryDataset
) -> tuple[DataLoader, ...]:

    train_loader = DataLoader(
        train_dataset,
        batch_size=param.batch_size,
        shuffle=True,
        pin_memory=False,
    )
    val_loader = DataLoader(val_dataset, batch_size=param.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=param.batch_size, shuffle=False)
    return train_loader, val_loader, test_loader


def _preprocess_data(
    data: dict[str, list],
    stats: Union[None, dict] = None
) -> tuple[dict[str, list], Union[None, dict]]:

    is_train = stats is None
    if is_train:
        stats = {
            "T_max": np.max([np.max(ti) for ti in data["t"]]),
            "y_max": np.max([np.max(yi) for yi in data["y"]]),
        }

    for i in range(len(data["t"])):
        # Normalize time grid.
        # data["t"][i] = data["t"][i].astype(np.float64) / stats["T_max"]

        # Normalize images.
        data["y"][i] = data["y"][i].astype(np.float32) / stats["y_max"]

        # Swap last two dimensions for compatibility with (S, M, N, D) dimensions.
        data["y"][i] = np.transpose(data["y"][i], (0, 2, 1))

    if is_train:
        return data, stats
    else:
        return data, None


def read_pickle(keys: list[str], path: str = "./") -> dict[str, ndarray]:
    data_dict = {}
    for key in keys:
        with open(path+key+".pkl", "rb") as f:
            data_dict[key] = pickle.load(f)
    return data_dict


def get_model_components(
    param: SimpleNamespace,
) -> tuple[NeuralDecoder, ODETransitionFunction, RecognitionNet]:

    nonlins = {
        "relu": nn.ReLU,
        "tanh": nn.Tanh,
        "gelu": nn.GELU,
        "mish": nn.Mish,
        "sine": Sine,
    }

    # Decoder.
    g = NeuralDecoder(
        decoder=nn.Sequential(
            CNNDecoder(param.K, param.N, param.D, 2, param.g_cnn_channels),
            ToNormalParameters(param.sigY),
        ),
    )

    # Transition function and recognition network.
    solver_kwargs = {
        "method": param.solver,
        "rtol": param.rtol,
        "atol": param.atol,
        "adjoint": param.adjoint,
        "options": {"step_size": 0.2},
    }
    if param.dyn_order == 1:
        F = ODETransitionFunction(
            f=nn.Sequential(
                nn.Linear(param.K, param.m_F*param.K), nonlins[param.F_nonlin](),
                nn.Linear(param.m_F*param.K, param.m_F*param.K), nonlins[param.F_nonlin](),
                nn.Linear(param.m_F*param.K, param.K)
            ),
            layers_to_count=[],
            solver_kwargs=solver_kwargs
        )
        h = RecognitionNet(
            phi_enc=CNNEncoder(param.m_h*param.K, param.N, param.D, param.h_enc_cnn_channels),
            phi_agg=create_agg_net(param, "static"),
            phi_gamma=nn.Linear(param.m_h*param.K, param.K),
            phi_tau=nn.Linear(param.m_h*param.K, param.K),
            tau_min=param.tau_min,
        )
    elif param.dyn_order == 2:
        assert param.K % 2 == 0, "Latent dimension `K` must be divisible by 2."
        F = ODETransitionFunctionSecondOrder(
            f=nn.Sequential(
                nn.Linear(param.K, param.m_F*param.K), nonlins[param.F_nonlin](),
                nn.Linear(param.m_F*param.K, param.m_F*param.K), nonlins[param.F_nonlin](),
                nn.Linear(param.m_F*param.K, param.K//2)
            ),
            layers_to_count=[],
            solver_kwargs=solver_kwargs
        )
        h = RecognitionNetSecondOrder(
            phi_enc=CNNEncoder(param.m_h*param.K, param.N, param.D, param.h_enc_cnn_channels),
            phi_agg=create_agg_net(param, "static"),
            phi_agg_dyn=create_agg_net(param, "dynamic"),
            phi_gamma=nn.Linear(param.m_h*param.K, param.K//2),
            phi_gamma_dyn=nn.Linear(param.m_h*param.K, param.K//2),
            phi_tau=nn.Linear(param.m_h*param.K, param.K//2),
            phi_tau_dyn=nn.Linear(param.m_h*param.K, param.K//2),
            tau_min=param.tau_min,
        )
    else:
        raise RuntimeError("Wrong dynamics order. Must be 1 or 2.")

    return g, F, h


def create_elbo(
    g: NeuralDecoder,
    F: ODETransitionFunction,
    h: RecognitionNet,
    param: SimpleNamespace
) -> AmortizedMultipleShootingELBO:

    prior_param_dict = nn.ParameterDict({
        "mu0": Parameter(0.0 * torch.ones([param.K]), False),
        "sig0": Parameter(1.0 * torch.ones([param.K]), False),
        "sigXi": Parameter(param.Xi / param.K**0.5 * torch.ones([1]), False),
        "mu_theta": Parameter(0.0 * torch.ones([1]), False),
        "sig_theta": Parameter(1.0 * torch.ones([1]), False),
    })
    posterior_param_dict = nn.ParameterDict({
        "mu_theta_g": Parameter(torch.cat([par.detach().reshape(-1) for par in g.parameters()])),
        "log_sig_theta_g": Parameter(-7.0 * torch.ones(g.param_count())),
        "mu_theta_F": Parameter(torch.cat([par.detach().reshape(-1) for par in F.parameters()])),
        "log_sig_theta_F": Parameter(-7.0 * torch.ones(F.param_count())),
    })
    if param.dyn_order == 1:
        p = ModelNormal(prior_param_dict, g, F)
    elif param.dyn_order == 2:
        p = ModelNormalSecondOrder(prior_param_dict, g, F)
    else:
        raise RuntimeError("Wrong dynamics order. Must be 1 or 2.")
    q = AmortizedMultipleShootingPosterior(posterior_param_dict, F, h)
    elbo = AmortizedMultipleShootingELBO(p, q)
    elbo.p.set_theta(elbo.q.sample_theta())
    return elbo


def visualize_trajectories(
    traj: list[ndarray],
    vis_inds: list[int],
    title: str,
    path: str,
    img_name: str,
) -> None:

    if not os.path.isdir(path):
        os.makedirs(path)

    img_size = 32
    panel_size = 5
    n_row = len(traj)
    n_col = len(vis_inds)

    fig, ax = plt.subplots(n_row, n_col, figsize=(panel_size*n_col, panel_size*n_row), squeeze=False)

    for i in range(n_row):
        for j in range(n_col):
            ax[i, j].imshow(traj[i][0, vis_inds[j], :, 0].reshape(img_size, img_size))  # type: ignore
            ax[i, j].grid(False)  # type: ignore
            # fig.colorbar(im, ax=ax[i, j], orientation='vertical')  # type: ignore

    fig.suptitle(title, fontsize=45)
    fig.tight_layout()
    plt.savefig(path+img_name)
    plt.close()


class ToNormalParameters(Module):
    """Converts output of CNNDecoder to parameters of p(y|x)."""
    def __init__(self, sigY) -> None:
        super().__init__()
        self.sigY = sigY

    def forward(self, x):
        x[..., 0] = torch.sigmoid(x[..., 0])  # to keep mean \in (0, 1)
        x[..., 1] = self.sigY  # fix standard deviation
        return x


def get_data_transform():
    transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.RandomHorizontalFlip(p=0.5),
        ]
    )

    def apply_transform(y: Tensor) -> Tensor:
        _, m, n, d = y.shape
        h, w = 32, 32
        y = rearrange(y, "s m (h w) d -> s (m d) h w", h=h, w=w)
        y = transform(y)
        y = rearrange(y, "s (m d) h w -> s m (h w) d", m=m, d=d)
        return y

    return apply_transform


def get_scheduler(optimizer, n_iters, lr):
    sched = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=(1e-5/lr)**(1.0/n_iters))
    return sched
