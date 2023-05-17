import matplotlib; matplotlib.use("TkAgg")
import torch
import torch.nn as nn
from Parameters import m, n

def is_symetric(mat, name="mat:", stage="unkown",iteration=0):
    symetric = bool(torch.allclose(mat, mat.T, rtol=1e-3, atol=1e-3))
    if not symetric:
        print(name+" is not stymmetric")
        print(mat)
        #raise ValueError(stage+" Stage: iteration: "+str(iteration) + " The "+name+" is not Symmetric.")
    return symetric

def is_psd(mat, name="mat:", stage="unkown",iteration=0):
    #symetric = bool(torch.allclose(mat, mat.T, rtol=1e-1, atol=1e-1))
    symetric = bool(torch.equal(mat, mat.T))
    psd = (bool((torch.linalg.eig(mat).eigenvalues.real[:]>=0).all()))
    if not psd:
        print(name+" is not positive semi definite")
        print(mat)
        print(name+ " eigenvalues:")
        print(torch.linalg.eig(mat).eigenvalues)
        string =stage+" Stage: iteration: "+str(iteration) + " The "+name+" is not positive semi-definite."
        #raise ValueError(string)
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
            m2x_posterior_batch[step],m2x_prior_batch[step],jac_H_batch[step], KG_batch[step] = environment.step(model = model, mode ="train_sequential", step=step)
            mse = estimation_mse_loss(environment.tgt_est_traj, environment.tgt_real_traj)
        return {"m2x_posterior":m2x_posterior_batch,"m2x_prior":m2x_prior_batch, "jac_H":jac_H_batch,"KG": KG_batch,"estimation_mse": mse}
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
            symetric = is_symetric(mat = mat,name = "single mode loss function before inverse and :3,:3")
            inf_theor_cost = (torch.inverse(args["m2x_prior"] -
                                            torch.matmul(args["KG"],
                                                         torch.matmul(args["jac_H"],
                                                                      args["m2x_prior"]))))[:3, :3]
            psd,symetric_2 = is_psd(mat = ((torch.inverse(inf_theor_cost))[:3, :3]), name =  "single mode loss function")

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

def plot_loss_steps(steps, losses):
    # Iterate over each list in losses
    for name, loss in losses:
        # Plot the data
        plt.plot(steps, loss, label=name)

    # Label the axes and add legend
    plt.xlabel('Number of Steps')
    plt.ylabel('Loss')
    plt.title('Loss vs Number of Steps')
    plt.legend()

    # Display the plot
    plt.show()

def plot_mse_steps(steps, mse):
    # Plot the data
    plt.plot(steps, mse)

    # Label the axes
    plt.xlabel('Number of Steps')
    plt.ylabel('MSE')
    plt.title('MSE vs Number of Steps')

    # Display the plot
    plt.show()