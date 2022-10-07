from types import SimpleNamespace

import torch.optim as optim

from tqdm import tqdm

import utils


param = {
    "T": 50,  # terminal time
    "M": 250,  # number of observations in [0, T]
    "sigY": 0.001,  # observation noise
    "seed": 1400,  # random seed

    "max_len": 201,  # truncation length for the trajectories

    "batch_size": 3,
    "lr": 0.01,  # learning rate
    "n_iters": 5000,  # number of optimization iterations

    "solver_kwargs": {"method": "rk4", "rtol": 1e-5, "atol": 1e-5, "adjoint": False},
}
param = SimpleNamespace(**param)

train_dataset = utils.create_datasets(param)
train_loader = utils.create_dataloaders(train_dataset, param)

utils.set_seed(param.seed)
g, F, _ = utils.get_model_components(param, construct_h=False)
elbo = utils.create_vms_elbo(g, F, param, S=len(train_dataset))

optimizer = optim.Adam(elbo.parameters(), lr=param.lr)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[4500], gamma=0.1)

for i in tqdm(range(param.n_iters), total=param.n_iters):
    t, y, traj_inds = next(iter(train_loader))
    L1, L2, L3, _, _ = elbo(t, y, traj_inds, block_size=10, scaler=1)
    loss = -(L1 - L2 - L3)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step()

print("Inferred parameter values =", elbo.q.posterior_param["mu_theta_F"][0:4])
print(f"True parameter values = {utils.LV_PARAM}")
