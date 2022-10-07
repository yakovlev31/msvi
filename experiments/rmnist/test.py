from types import SimpleNamespace

import torch

import wandb
from tqdm import tqdm

from einops import reduce

import msvi.utils.rmnist as data_utils

import utils


torch.backends.cudnn.benchmark = True  # type: ignore


# Read parameters.
argparser = data_utils.create_argparser()
param = SimpleNamespace(**vars(argparser.parse_args()))
param.tags.append("test")

# Load data.
train_dataset, val_dataset, test_dataset = data_utils.create_datasets(param)
train_loader, val_loader, test_loader = data_utils.create_dataloaders(param, train_dataset, val_dataset, test_dataset)


# Create and load model.
utils.set_seed(param.seed)
device = torch.device(param.device)
g, F, h = data_utils.get_model_components(param)
elbo = data_utils.create_elbo(g, F, h, param).to(device)
utils.load_model(elbo, param.model_folder, param.name, device)
elbo.eval()

wandb.init(
    mode="disabled",  # online/disabled
    project="AVMS",
    group=param.group,
    tags=param.tags,
    name=param.name,
    config=vars(param),
    save_code=True,
)

loss_fn = torch.nn.MSELoss(reduction="none")

with torch.no_grad():
    losses = []
    for batch in tqdm(test_loader, total=len(test_loader)):
        t, y, traj_inds = [bi.to(device) for bi in batch]

        t_inf, y_inf = utils.get_inference_data(t, y, param.delta_inf)

        y_pd = torch.zeros_like(y)
        for i in range(param.n_mc_samples):
            x0 = utils.get_x0(elbo, t_inf, y_inf)
            y_pd += utils._pred_full_traj(elbo, t, x0)
        y_pd /= param.n_mc_samples

        loss_per_traj = reduce(loss_fn(y_pd, y), "s m n d -> s () () ()", "mean").view(-1).detach().cpu().numpy().ravel()
        losses.extend(loss_per_traj)

mean_loss = sum(losses) / len(losses)
print(mean_loss)

wandb.run.summary.update({"mean_test_loss": mean_loss})  # type: ignore
