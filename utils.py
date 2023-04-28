import numpy as np
import matplotlib; matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation

def update_graph(num, data, real_traj, tracker_traj, tail,dot, real_tale, real_dot, tracker_tale, tracker_dot):
    if num >= 100:
        tail.set_data(data[num-50:num-1, 0], data[num-50:num-1, 1])
        tail.set_3d_properties(data[num-50:num-1, 2])

        real_tale.set_data(real_traj[num-50:num-1, 0], real_traj[num-50:num-1, 1])
        real_tale.set_3d_properties(real_traj[num-50:num-1, 2])

        tracker_tale.set_data(tracker_traj[num-50:num-1, 0], tracker_traj[num-50:num-1, 1])
        tracker_tale.set_3d_properties(tracker_traj[num - 50:num - 1, 2])
    elif num >= 1:
        tail.set_data(data[:num-1, 0], data[:num-1, 1])
        tail.set_3d_properties(data[:num-1, 2])

        real_tale.set_data(real_traj[:num-1, 0], real_traj[:num-1, 1])
        real_tale.set_3d_properties(real_traj[:num-1, 2])

        tracker_tale.set_data(tracker_traj[:num-1, 0], tracker_traj[:num-1, 1])
        tracker_tale.set_3d_properties(tracker_traj[:num-1, 2])

    dot.set_data(data[num, 0], data[num, 1])
    dot.set_3d_properties(data[num, 2])
    #dot.set_color('b')

    real_dot.set_data(real_traj[num, 0], real_traj[num, 1])
    real_dot.set_3d_properties(real_traj[num, 2])
    #real_dot.set_color('g')
    tracker_dot.set_data(tracker_traj[num, 0], tracker_traj[num, 1])
    tracker_dot.set_3d_properties(tracker_traj[num, 2])
    #tail.set_color('b')
    #real_tale.set_color('m')
    return tail,dot, real_tale, real_dot,tracker_tale,tracker_dot

def calculate_mse(a, b, warmup_steps=5):
    if len(a) != len(b):
        raise ValueError("Both input arrays must have the same length")

    mse = np.mean((a[warmup_steps:] - b[warmup_steps:]) ** 2)
    return mse