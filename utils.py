import numpy as np
import matplotlib; matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
import torch
def update_graph(num, tgt_est_traj, real_traj, tracker_traj, est_tail, est_dot, real_tale, real_dot, tracker_tale, tracker_dot):
    length_mx = 0
    length_mn = 0
    if num>1:
        length_mx = num
        length_mn =1
        if num >= 100:
            length_mx = 50
            length_mn= 1
        x = tgt_est_traj[num - length_mx:num-length_mn , 0,0].numpy()
        y = tgt_est_traj[num - length_mx:num-length_mn, 1,0].numpy()
        z = tgt_est_traj[num - length_mx:num-length_mn, 2,0].numpy()
        est_tail.set_data(x, y)
        est_tail.set_3d_properties(z)

        x =real_traj[num - length_mx:num-length_mn, 0, 0].numpy()
        y =real_traj[num - length_mx:num-length_mn, 1, 0].numpy()
        z =real_traj[num - length_mx:num-length_mn, 2, 0].numpy()
        real_tale.set_data(x, y)
        real_tale.set_3d_properties(z)

        x = tracker_traj[num - length_mx:num-length_mn, 0, 0].numpy()
        y = tracker_traj[num - length_mx:num-length_mn, 1, 0].numpy()
        z = tracker_traj[num - length_mx:num-length_mn, 2, 0].numpy()
        tracker_tale.set_data(x, y)
        tracker_tale.set_3d_properties(z)
    else:
        pass

    est_dot.set_data(tgt_est_traj[num, 0,0].item(), tgt_est_traj[num, 1,0].item())
    est_dot.set_3d_properties(tgt_est_traj[num, 2,0].item())

    real_dot.set_data(real_traj[num, 0,0], real_traj[num, 1,0].item())
    real_dot.set_3d_properties(real_traj[num, 2,0].item())

    tracker_dot.set_data(tracker_traj[num, 0].item(), tracker_traj[num, 1,0].item())
    tracker_dot.set_3d_properties(tracker_traj[num, 2,0].item())

    return est_tail,est_dot, real_tale, real_dot,tracker_tale,tracker_dot

def calculate_loss(tensor1, tensor2, warmup_steps=0, type ="mse"):
    if len(tensor1) != len(tensor2):
        raise ValueError("Both input arrays must have the same length")
    if type == "mse":
        # Calculate the MSE between the two tensors using broadcasting
        mse_loss = torch.nn.MSELoss(reduction='mean')
        mse = mse_loss(tensor1, tensor2)
        print("MSE:", mse.item())
        return mse
    else:
        pass