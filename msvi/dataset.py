from typing import Union
import torch
from torch.utils.data import Dataset


Tensor = torch.Tensor


class TrajectoryDataset(Dataset):
    """Stores trajectories and time grids.

    Used to store trajectories `y` and the corresponding time grids `t`.
    Each trajectory is assumed to have three dimensions:
        (time points, observation points, observation dim.).
    Each time grid is assimed to have two dimensions: (time points, 1).
    If `max_len` is not None, a subtrajectory of length `max_len` is
        selected from each trajectory and time grid.

    Args:
        t: Contains time grids of various lengths M.
            The shape of each time grid t[i] must be (M_i, 1).
        y: Contrains trajectories of various lengths.
            The shape of each trajectory y[i] must be (M_i, N, D).
        max_len: Length of subtrajectories selected from each trajectory and time grid.
    """
    def __init__(self, t: list[Tensor], y: list[Tensor], max_len: Union[None, int] = None) -> None:
        self.t = t
        self.y = y
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.t)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor, Tensor]:
        t = self.t[idx]
        y = self.y[idx]
        traj_ind = torch.tensor(idx, dtype=torch.long)

        if self.max_len is not None:
            t = t[:self.max_len]
            y = y[:self.max_len]

        return t, y, traj_ind
