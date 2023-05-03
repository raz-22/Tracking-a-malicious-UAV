import numpy as np
import matplotlib; matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
import torch

def calculate_loss(tgt_est_traj, tgt_real_traj):
    """
    Calculate the mean squared error (MSE) between the estimated and real trajectories.

    Args:
    tgt_est_traj (np.array): The estimated trajectory of the target (N x 3).
    tgt_real_traj (np.array): The real trajectory of the target (N x 3).

    Returns:
    float: The mean squared error between the estimated and real trajectories.
    """
    if tgt_est_traj.shape != tgt_real_traj.shape:
        raise ValueError("The shapes of the estimated and real trajectories must match.")

    mse = torch.mean((tgt_est_traj - tgt_real_traj)**2).item()
    return mse