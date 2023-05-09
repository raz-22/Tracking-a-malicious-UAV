import numpy as np
import matplotlib; matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
import torch
import torch.nn as nn
from parameters import m, n


def is_psd(mat):
    return bool((mat == mat.T).all() and (torch.eig(mat)[0][:, 0] >= 0).all())
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
    mse = torch.mean(torch.square(tgt_est_traj - tgt_real_traj)).item()
    return mse
def generate_traj(environment,model, num_steps,mode = "sequential"):
    if mode == "sequential":
        m2x_prior_batch = torch.zeros(num_steps,m,m)
        m2x_posterior_batch = torch.zeros(num_steps,m,m)
        jac_H_batch = torch.zeros(num_steps,n,m)
        KG_batch = torch.zeros(num_steps,m,n)
        running_loss=0
        for step in range(num_steps):
            m2x_posterior ,m2x_prior, jac_H, KG = environment.step(model = model, mode ="train_sequential")
            m2x_posterior_batch[step, :, :] = m2x_posterior
            m2x_prior_batch[step,:,:] = m2x_prior
            jac_H_batch[step,:, :] = jac_H
            KG_batch[step,:, :] = KG

            #Fixme: temporary test to see batch loss equals to sequential calculation
            loss = information_theoretic_cost({"m2x_prior":m2x_prior,"jac_H":jac_H,"KG":KG}, mode= "single")
            if torch.isnan(loss):
                print("information determinant is negative")
            running_loss += loss.item()
        return {"m2x_posterior":m2x_posterior_batch,"m2x_prior":m2x_prior_batch, "jac_H":jac_H_batch,"KG": KG_batch}, (running_loss/num_steps)
    if mode == "batch_sequential":
        pass
def information_theoretic_cost(args,mode = "single"):
    if mode =="single":
        inf_theor_cost = (torch.inverse(args["m2x_prior"] -
                                        torch.matmul(args["KG"],
                                                     torch.matmul(args["jac_H"],
                                                                  args["m2x_prior"]))))[:3,:3]
        """Calculates -ln(det(matrix))"""
        det = torch.det(inf_theor_cost)
        if det<0:
            print("determinant is negative")
        ln_det_cost = -torch.log(det)
        return ln_det_cost

class InfromationTheoreticCost(nn.Module):
    def __init__(self, weight=1):
        super(InfromationTheoreticCost, self).__init__()
        self.weight = weight

    def forward(self, args,mode = "single"):
        if mode == "single":
            inf_theor_cost = (torch.inverse(args["m2x_prior"] -
                                            torch.matmul(args["KG"],
                                                         torch.matmul(args["jac_H"],
                                                                      args["m2x_prior"]))))[:3, :3]
            """Calculates -ln(det(matrix))"""
            det = torch.det(inf_theor_cost)
            if det < 0:
                print("shit")
            ln_det_cost = -torch.log(det)
            return ln_det_cost
        if mode == "sequential":
            """Calculates -ln(det(matrix))"""
            inf_theor_cost = (torch.inverse(args["m2x_posterior"]))[:,:3, :3]
            det = torch.linalg.det(inf_theor_cost)
            ln_det_cost = -torch.log(det)
            loss = torch.mean(ln_det_cost)
            return loss
            # inf_theor_cost = (torch.inverse(args["m2x_prior"] -
            #                                 torch.bmm(args["KG"],
            #                                              torch.bmm(args["jac_H"],
            #                                                           args["m2x_prior"]))))[:,:3, :3]
            # """Calculates -ln(det(matrix))"""
            # det = torch.linalg.det(inf_theor_cost)
            # ln_det_cost = -torch.log(det)
            # loss = torch.mean(ln_det_cost)
            # return loss
        if mode == "batch_sequential":
            pass