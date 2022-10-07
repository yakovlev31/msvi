from types import SimpleNamespace

import torch.optim as optim

from tqdm import tqdm

import utils


param = {
    "T": 50,  # terminal time
    "M": 250,  # number of observations in [0, T]
    "sigY": 0.001,  # observation noise
    "max_len": 201,  # truncation length for the trajectories

    "seed": 1400,  # random seed

    "batch_size": 3,
    "lr": 0.01,  # learning rate
    "n_iters": 5000,  # number of optimization iterations

    "solver_kwargs": {"method": "rk4", "rtol": 1e-5, "atol": 1e-5, "adjoint": False},

    # Parameters for recognition network.
    "h_agg_attn": "tdp",
    "h_agg_pos_enc": "rpeNN",
    "h_agg_stat_layers": 2,
    "K": 2,
    "m_h": 16,
    "h_agg_max_tokens": 500,
    "h_agg_max_time": 100,
    "h_agg_delta_r": 10,
    "h_agg_p": -1,
    "n": 1,
    "drop_prob": 0,

    "block_size": 1,
}
param = SimpleNamespace(**param)

train_dataset = utils.create_datasets(param)
train_loader = utils.create_dataloaders(train_dataset, param)

utils.set_seed(param.seed)
g, F, h = utils.get_model_components(param, construct_h=True)
elbo = utils.create_avms_elbo(g, F, h, param)

optimizer = optim.Adam(elbo.parameters(), lr=param.lr)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[4500], gamma=0.1)

for i in tqdm(range(param.n_iters), total=param.n_iters):
    t, y, traj_inds = next(iter(train_loader))
    elbo.q.rec_net.update_time_grids(t)

    L1, L2, L3, _, _ = elbo(t, y, traj_inds, block_size=param.block_size, scaler=1)
    loss = -(L1 - L2 - L3)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step()

print("Inferred parameter values =", elbo.q.posterior_param["mu_theta_F"][0:4])
print(f"True parameter values = {utils.LV_PARAM}")
