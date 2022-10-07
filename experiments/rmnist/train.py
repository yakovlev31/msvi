from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.optim as optim

import wandb
from tqdm import tqdm

import msvi.utils.rmnist as data_utils

import utils


torch.backends.cudnn.benchmark = True  # type: ignore


# Read parameters.
argparser = data_utils.create_argparser()
param = SimpleNamespace(**vars(argparser.parse_args()))
param.tags.append("train")


# Load data.
train_dataset, val_dataset, _ = data_utils.create_datasets(param)
train_loader, val_loader, _ = data_utils.create_dataloaders(param, train_dataset, val_dataset, val_dataset)


# Create model.
utils.set_seed(param.seed)
device = torch.device(param.device)
g, F, h = data_utils.get_model_components(param)
elbo = data_utils.create_elbo(g, F, h, param).to(device)


# Training.
optimizer = optim.Adam(elbo.parameters(), lr=param.lr)
scheduler = data_utils.get_scheduler(optimizer, param.n_iters, param.lr)

bma = utils.BatchMovingAverage(k=10)
data_transform = data_utils.get_data_transform()

wandb.init(
    mode="disabled",  # online/disabled
    project="AVMS",
    group=param.group,
    tags=param.tags,
    name=param.name,
    config=vars(param),
    save_code=True,
)

utils.set_seed(param.seed)
for i in tqdm(range(param.n_iters), total=param.n_iters):
    elbo.train()
    t, y, traj_inds = [bi.to(device) for bi in next(iter(train_loader))]

    # t = t + (torch.rand_like(t) - 0.5) * 2 * param.sigT
    y = data_transform(y)

    L1, L2, L3, x, s = elbo(t, y, traj_inds, param.block_size, scaler=1.0)
    L1 *= len(train_dataset) / param.batch_size
    L2 *= len(train_dataset) / param.batch_size
    loss = -(L1 - L2 - L3)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step()

    # Validation on full trajectory predictions.
    if i % int(0.00333 * param.n_iters) == 0 or i == param.n_iters - 1:
        with torch.no_grad():
            elbo.eval()
            t_val, y_val, _ = [bi.to(device) for bi in next(iter(val_loader))]

            y_full_traj = utils.pred_full_traj(param, elbo, t, y)
            y_val_full_traj = utils.pred_full_traj(param, elbo, t_val, y_val)

            train_full_traj_mse = nn.MSELoss()(y_full_traj, y).item()
            val_full_traj_mse = nn.MSELoss()(y_val_full_traj, y_val).item()

            bma.add_value(val_full_traj_mse)
            if bma.get_average() <= bma.get_min_average():
                utils.save_model(elbo, param.model_folder, param.name)

            wandb.log(
                {
                    "-L1": -L1.item(),
                    "L2": L2.item(),
                    "L3": L3.item(),
                    "-ELBO": loss.item(),

                    "train_full_traj_mse": train_full_traj_mse,
                    "val_full_traj_mse": val_full_traj_mse,

                    "lr": optimizer.param_groups[0]["lr"],
                    "scaler": 1.0,
                },
                step=i
            )

            if param.visualize == 1:
                data_utils.visualize_trajectories(
                    traj=[
                        y[[0]].detach().cpu().numpy(),
                        y_full_traj[[0]].detach().cpu().numpy(),
                        y_val[[0]].detach().cpu().numpy(),
                        y_val_full_traj[[0]].detach().cpu().numpy(),
                    ],
                    vis_inds=list(range(y.shape[1]))[:-1:max(1, int(0.09*y.shape[1]))],
                    title=f"Iteration {i}",
                    path=f"./img/{param.name}/",
                    img_name=f"iter_{i}.png",
                )
