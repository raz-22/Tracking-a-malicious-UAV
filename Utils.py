import matplotlib; matplotlib.use("TkAgg")
import torch
import torch.nn as nn
from Parameters import m, n

def is_symetric(mat):
    symetric = bool(torch.allclose(mat, mat.T, rtol=1e-3, atol=1e-3))
    return symetric

def is_psd(mat):
    symetric = bool(torch.allclose(mat, mat.T, rtol=1e-1, atol=1e-1))
    psd = (bool((torch.linalg.eig(mat).eigenvalues.real[:]>=0).all()))
    return psd, symetric


def generate_traj(environment,model, num_steps,mode = "sequential"):
    if mode == "sequential":
        m2x_prior_batch, m2x_posterior_batch, jac_H_batch, KG_batch = [torch.zeros(num_steps, *size) for size in
                                                                       [(m, m), (m, m), (n, m), (m, n)]]
        m2x_prior_batch = torch.zeros(num_steps,m,m)
        m2x_posterior_batch = torch.zeros(num_steps,m,m)
        jac_H_batch = torch.zeros(num_steps,n,m)
        KG_batch = torch.zeros(num_steps,m,n)
        running_loss=0
        for step in range(num_steps):
            m2x_posterior_batch[step],m2x_prior_batch[step],
            jac_H_batch[step], KG_batch[step] = environment.step(model = model, mode ="train_sequential", step=step)
        return {"m2x_posterior":m2x_posterior_batch,"m2x_prior":m2x_prior_batch, "jac_H":jac_H_batch,"KG": KG_batch}
    if mode == "batch_sequential":
        pass

class InfromationTheoreticCost(nn.Module):
    def __init__(self, weight=1):
        super(InfromationTheoreticCost, self).__init__()
        self.weight = weight

    def forward(self, args,mode = "single"):
        if mode == "single":
            mat = args["m2x_prior"] -torch.matmul(args["KG"],
                                                         torch.matmul(args["jac_H"],
                                                                      args["m2x_prior"]))
            symetric = is_symetric(mat)
            inf_theor_cost = (torch.inverse(args["m2x_prior"] -
                                            torch.matmul(args["KG"],
                                                         torch.matmul(args["jac_H"],
                                                                      args["m2x_prior"]))))[:3, :3]
            psd,symetric_2 = is_psd((torch.inverse(inf_theor_cost))[:3, :3])

            # Print result
            if not symetric:
                raise ValueError("Single Mode Cost:the 6X6 cost is not symetric")
            if not symetric_2:
                raise ValueError("Single Mode Cost:The prior 3X3 cost is not symetric")
            if not psd:
                raise ValueError("Single Mode Cost:The prior 3X3 cost is not psd")
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

        if mode == "batch_sequential":
            pass

def estimation_mse_loss(tgt_est_traj, tgt_real_traj):
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
    return torch.mean(torch.square(tgt_est_traj - tgt_real_traj)).item()