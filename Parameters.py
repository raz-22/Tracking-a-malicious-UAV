
import torch
from torch import autograd

def is_symetric(mat):
    symetric = bool(torch.allclose(mat, mat.T, rtol=1e-1, atol=1e-1))
    return symetric

def is_psd(mat):
    symetric = bool(torch.allclose(mat, mat.T, rtol=1e-1, atol=1e-1))
    psd = (bool((torch.linalg.eig(mat).eigenvalues.real[:]>=0).all()))
    return psd, symetric
torch.pi = torch.acos(torch.zeros(1)).item() * 2 # which is 3.1415927410125732
#########################
### Design Parameters ###
#########################

### State & Observation Vectors size ###
m = 6
n = 4
### Number Of Time Steps & Batch Size ###

T=1
batch_size = 1

### initial Tracker State ###
tracker_state = torch.reshape(torch.tensor([[50], [50], [80], [-0.3], [0.4], [1e-5]]), (m,1))

### Initial State 1ST and 2ND Moments ###
m1x_0 = torch.reshape(torch.tensor([[1e-5], [1e-5], [90], [-0.3], [0.4], [1e-5]]), (m,1))
m2x_0 = torch.eye(m) * torch.tensor([20**2, 20**2, 20**2, 0.5**2, 0.5**2, 0.5**2])
if is_symetric(m2x_0)== False:
    raise ValueError('Initial 2nd Moment Is Not Symetric')
if is_psd(m2x_0)[0] == False:
    raise ValueError("Initial 2nd Moment Is Not Positive Semy Definite")


### Radar Constants from the Paper ###

gamma = 4
lamda = 3.8961*(1e-3) # estimated for 77ghz frequency
delta_t = 1

### Initial State & Observation Noise Covariance ###
def Init_Cov_Matrix():
    # Calculating the noise covariance matrix , constants are from the use case in the original paper
    diagonal = [1e-5, 1e-5, 1e-6]
    diagonal_matrix = torch.diag(torch.tensor(diagonal))
    A = ((delta_t ** 3) / 3) * diagonal_matrix
    B = ((delta_t ** 2) / 2) * diagonal_matrix
    C = ((delta_t ** 2) / 2) * diagonal_matrix
    D = delta_t * diagonal_matrix
    top = torch.cat((A, B), dim=1)
    bottom = torch.cat((C, D), dim=1)
    Q = torch.cat((top, bottom), dim=0)
    diagonal = [1e-5, 1e-5, 1e-6, 1e-6]
    diagonal_matrix = torch.diag(torch.tensor(diagonal))
    R = torch.eye(n) * diagonal_matrix
    if is_symetric(R) == False:
        if is_psd(R)[0] == False:
            raise ValueError("Initial Observation Noise Covariance Is Not Positive Semy Definite")
        else:
            raise ValueError('Initial Observation Noise Covariance Is Not Symetric')
    if is_psd(R)[0] == False:
        raise ValueError('Initial Observation Noise Covariance Is Not Symetric')
        if is_symetric(Q) == False:
            if is_psd(Q)[0] == False:
                raise ValueError("Initial State Noise Covariance Is Not Positive Semy Definite")
            else:
                raise ValueError('Initial State Noise Covariance Is Not Symetric')
    if is_psd(Q)[0] == False:
        raise ValueError('Initial Observation Noise Covariance Is Not Symetric')

    return Q, R

###############################
### Dynamic Model Equations ###
###############################
# Calculating the state matrix
A = torch.eye(3)
B = torch.eye(3) * delta_t
C = torch.zeros((3, 3))
D = torch.eye(3)
top = torch.cat((A, B), dim=1)
bottom = torch.cat((C, D), dim=1)
state_mat = torch.cat((top, bottom), dim=0)
print(state_mat)
if is_psd(state_mat)[0] == False:
    raise ValueError("state_mat Is Not Positive Semy Definite")

### State Evolution Function ###
def f(x, jacobian=False):
    F = state_mat
    if jacobian:
        return torch.matmul(F, x), F
    else:
        return torch.matmul(F, x)

### State Observation Function ###
def obs(x_new):
    h0 = x_new[0] ** 2 + x_new[1]
    h1 = x_new[2]
    h2 = x_new[0] + 2 * x_new[3] ** 3
    h3 = x_new[4] + x_new[5]
    return  torch.stack([h0, h1, h2,h3], dim=0)

def h(x,tracker_state,jacobian=False, type="dummy"):
    # Define x as a tensor with requires_grad=True to enable autograd
    x_new = x.clone().detach().requires_grad_(True)
    # Define h_x as a tensor
    h_x = obs(x_new)
    # J  should be =
    #   2x_0        1          0          0          0          0
    #    0          0          1          0          0          0
    #    1          0          0       3x_3**2       0          0
    #    0          0          0          0          1          1

    # Define the dimensions of h_x and the Jacobian matrix
    n = h_x.shape[0]
    m = x_new.shape[0]
    # Define the Jacobian matrix if jacobian=True
    if jacobian:
        x_new.grad = None
        J = torch.autograd.functional.jacobian(obs, x_new, create_graph=True)
        h_x = torch.reshape(h_x[:], (n, 1))
        J = J.view(n, m)
        return h_x, J
    else:
        h_x = torch.reshape(h_x[:], (n, 1))
        return h_x

def real_h(x,tracker_state,jacobian=False, type="dummy"):
    los = x[:3, 0] - tracker_state[:3, 0]
    dv = x[3:, 0] - tracker_state[3:, 0]
    delta_x = los[0]
    delta_y = los[1]
    delta_z = los[2]

    # h(s) observstion function measurement equations
    d = torch.norm(los)
    if d.item() == 0:
        print('zero')
    gamma_d_2 = (gamma / 2) * (d)
    # if delta_x ==torch.tensor(0):
    #     if delta_y >0 :
    #         azimuth = (torch.pi)/2
    #         elevation = torch.atan(delta_z / d)
    #         radian_velocity = torch.dot(los, dv) / d
    #         doppler_shift = (gamma * radian_velocity) / (2 * lamda)
    #     else:
    #         azimuth = (torch.pi)*1.5
    #         elevation = torch.atan(delta_z / d)
    #         radian_velocity = torch.dot(los, dv) / d
    #         doppler_shift = (gamma * radian_velocity) / (2 * lamda)
#else:
    azimuth = torch.atan(delta_y / delta_x)
    elevation = torch.acos(delta_z / d)
    radian_velocity = torch.dot(los, dv) / d
    doppler_shift = (gamma * radian_velocity) / (2 * lamda)
    if torch.isnan(gamma_d_2.any()) or torch.isnan(azimuth.any()) or torch.isnan(elevation.any()) or torch.isnan(doppler_shift.any()):
        raise ValueError("one of the observations is nan")
    # if d == torch.tensor(0):
    #     azimuth = torch.tensor(0)
    #     elevation = torch.tensor(0)
    #     doppler_shift = torch.tensor(0)
    #     radian_velocity = torch.tensor(0)

    o = torch.stack((gamma_d_2, azimuth, elevation, doppler_shift))
    o = torch.reshape(o[:], (o.shape[0],1))
    if jacobian:
        jac_h =autograd_h(x,tracker_state)
        return o, jac_h
    else:
        return o


def autograd_h(x, tracker_state):
    # Define the function for which the Jacobian will be computed
    def h(x):
        los = x[:3, 0] - tracker_state[:3, 0]
        dv = x[3:, 0] - tracker_state[3:, 0]
        delta_x = los[0]
        delta_y = los[1]
        delta_z = los[2]

        # h(s) observstion function measurement equations
        d = torch.norm(los)
        if d.item() == 0:
            print('zero')
        gamma_d_2 = (gamma / 2) * (d)
        azimuth = torch.atan(delta_y / delta_x)
        elevation = torch.acos(delta_z / d)
        radian_velocity = torch.dot(los, dv) / d
        doppler_shift = (gamma * radian_velocity) / (2 * lamda)
        if torch.isnan(gamma_d_2.any()) or torch.isnan(azimuth.any()) or torch.isnan(elevation.any()) or torch.isnan(doppler_shift.any()):
            raise ValueError("one of the observations is nan")
        o = torch.stack((gamma_d_2, azimuth, elevation, doppler_shift))
        o = torch.reshape(o[:], (n, 1))
        return o

    # Compute the Jacobian of h with respect to x
    jac_h = torch.autograd.functional.jacobian(h, x).view(n,m)
    return jac_h


##################################
### Utils for non-linear cases ###
##################################
def getJacobian(x, tracker_state, g):
    """
    Currently, pytorch does not have a built-in function to compute Jacobian matrix
    in a batched manner, so we have to iterate over the batch dimension.

    input x (torch.tensor): [batch_size, m/n, 1]
    input g (function): function to be differentiated
    output Jac (torch.tensor): [batch_size, m, m] for f, [batch_size, n, m] for h
    """
    # Method 1: using autograd.functional.jacobian
    # batch_size = x.shape[0]
    # Jac_x0 = torch.squeeze(autograd.functional.jacobian(g, torch.unsqueeze(x[0,:,:],0)))
    # Jac = torch.zeros([batch_size, Jac_x0.shape[0], Jac_x0.shape[1]])
    # Jac[0,:,:] = Jac_x0
    # for i in range(1,batch_size):
    #     Jac[i,:,:] = torch.squeeze(autograd.functional.jacobian(g, torch.unsqueeze(x[i,:,:],0)))
    # Method 2: using F, H directly
    if g == h:
        _, Jac = g(x, tracker_state, jacobian=True)
    else:
        _, Jac = g(x, jacobian=True)
    return Jac
