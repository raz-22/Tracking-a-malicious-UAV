import numpy as np
import matplotlib; matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation

def update_graph(num, coords, real_traj, ax):
    ax.clear()
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    if num > 0:
        ax.plot(coords[:num, 0], coords[:num, 1], coords[:num, 2], lw=2, c='r', label='current_position')
        ax.plot(real_traj[:num, 0], real_traj[:num, 1], real_traj[:num, 2], lw=2, c='b', label='real_traj')

    if num == 1:
        ax.legend()

    return ax

def calculate_mse(a, b, warmup_steps=5):
    if len(a) != len(b):
        raise ValueError("Both input arrays must have the same length")

    mse = np.mean((a[warmup_steps:] - b[warmup_steps:]) ** 2)
    return mse