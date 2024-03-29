"""This file contains the parameters for the Lorenz Atractor simulation.

Update 2023-02-06: f and h support batch size speed up

"""


import torch
import numpy as np

import math
torch.pi = torch.acos(torch.zeros(1)).item() * 2 # which is 3.1415927410125732
from torch import autograd

#########################
### Design Parameters ###
#########################
m = 6
n = 4
# TODO: UPDATE THESE PARAMETERS
variance = 0
m1x_0 = torch.reshape(torch.tensor([[1e-5], [1e-5], [90], [-0.3], [0.4], [1e-5]]), (m,1))

diagonal_values = [20**2, 20**2, 20**2, 0.5**2, 0.5**2, 0.5**2]
diagonal_tensor = torch.tensor(diagonal_values)
m2x_0 = torch.eye(m) * diagonal_tensor


###Observation Function Parameters
gamma = 4
lamda = 3.8961*(1e-3) # estimated for 77ghz frequency

delta_t = 1
######################################################
### State evolution function f for  UAV usecase ###
######################################################
target_state = torch.reshape(torch.tensor([[1e-5], [1e-5], [90], [-0.3], [0.4], [1e-5]]), (m,1))
tracker_state = torch.reshape(torch.tensor([[50], [50], [80], [-0.3], [0.4], [1e-5]]), (m,1))
if target_state.any() != m1x_0.any():
    raise ValueError("m1x_0 and initial tracker state must be equal")

### f will be fed to filters and KNet, note that the mismatch comes from delta_t
def f(x, jacobian=False):
    """
    BX = torch.zeros([x.shape[0],m,m]).float() #[batch_size, m, m]
    BX[:,1,0] = torch.squeeze(-x[:,2,:]) 
    BX[:,2,0] = torch.squeeze(x[:,1,:]) 
    Const = C
    A = torch.add(BX, Const) 
    # Taylor Expansion for F    
    F = torch.eye(m)
    F = F.reshape((1, m, m)).repeat(x.shape[0], 1, 1) # [batch_size, m, m] identity matrix
    for j in range(1,J+1):
        F_add = (torch.matrix_power(A*delta_t, j)/math.factorial(j))
        F = torch.add(F, F_add)
    """
    # Calculating the state matrix
    A = torch.eye(3)
    B = torch.eye(3) * delta_t
    C = torch.zeros((3, 3))
    D = torch.eye(3)
    top = torch.cat((A, B), dim=1)
    bottom = torch.cat((C, D), dim=1)
    state_mat = torch.cat((top, bottom), dim=0)
    F = state_mat

    if jacobian:
        return torch.matmul(F, x), F
    else:
        return torch.matmul(F, x)

##################################################
### Observation function h for  UAV usecase ###
##################################################
def obs(x_new):
    h0 = x_new[0] ** 2 + x_new[1]
    h1 = x_new[2]
    h2 = x_new[0] + 2 * x_new[3] ** 3
    h3 = x_new[4] + x_new[5]
    return  torch.stack([h0, h1, h2,h3], dim=0)
########################
######DUMMY H###########
########################
def h(x,tracker_state,jacobian=False):
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

def true_h(x,tracker_state,jacobian=False):
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
    if delta_x ==torch.tensor(0):
        if delta_y >0 :
            azimuth = (torch.pi)/2
            elevation = torch.atan(delta_z / d)
            # TODO: validate the radian velocity formula
            radian_velocity = torch.dot(los, dv) / d
            doppler_shift = (gamma * radian_velocity) / (2 * lamda)
        else:
            azimuth = (torch.pi)*1.5
            elevation = torch.atan(delta_z / d)
            # TODO: validate the radian velocity formula
            radian_velocity = torch.dot(los, dv) / d
            doppler_shift = (gamma * radian_velocity) / (2 * lamda)
    else:
        azimuth = torch.atan(delta_y / delta_x)
        elevation = torch.atan(delta_z / d)
        # TODO: validate the radian velocity formula
        radian_velocity = torch.dot(los, dv) / d
        doppler_shift = (gamma * radian_velocity) / (2 * lamda)
    if d == torch.tensor(0):
        azimuth = torch.tensor(0)
        elevation = torch.tensor(0)
        doppler_shift = torch.tensor(0)
        radian_velocity = torch.tensor(0)

    o = torch.stack((gamma_d_2, azimuth, elevation, doppler_shift))
    o = torch.reshape(o[:], (n,1))
    if jacobian:

       # arg_1 = azimuth
       # arg_2 = elevation
       #  a = torch.stack(
       #      [(torch.cos(arg_1) * torch.sin(arg_2)), (torch.sin(arg_1) * torch.sin(arg_2)), torch.cos(arg_2)])
       #  a = torch.reshape(a[:], (1, 3))
       #
       #
       #
       #  #todo: breakpoint
       #  jac_h_11 = (gamma/2)*a
       #  jac_h_42 = (gamma / (2 * lamda)) * a
       #
       #  arg_1 = (azimuth+(torch.pi/2))
       #  arg_2 = torch.tensor([0.5])*torch.pi
       #
       #  a = torch.stack(
       #      [(torch.cos(arg_1) * torch.sin(arg_2)), (torch.sin(arg_1) * torch.sin(arg_2)), torch.cos(arg_2)])
       #  a = torch.reshape(a[:], (1, 3))
       #  jac_h_21 = a/(d*torch.sin(elevation))
       #
       #  arg_1 = azimuth
       #  arg_2 = elevation+(torch.pi / 2)
       #  a = torch.stack(
       #      [(torch.cos(arg_1) * torch.sin(arg_2)), (torch.sin(arg_1) * torch.sin(arg_2)), torch.cos(arg_2)])
       #  a = torch.reshape(a[:], (1, 3))
       #  jac_h_31 = a/d
       #
       #  w = torch.cross(los,dv)/(d**2)
       #  w = torch.reshape(w[:], (1, 3))
       #  jac_h_41_1 = (gamma/(2*lamda))*((torch.cos(elevation)*w[0][1])-(torch.sin(azimuth)*torch.sin(elevation)*w[0][2]))
       #  jac_h_41_2 = (gamma / (2 * lamda)) * ((torch.cos(azimuth)*torch.sin(elevation)*w[0][2])-(torch.cos(elevation)*w[0][0]))
       #  jac_h_41_3 = (gamma / (2 * lamda)) * ((torch.sin(azimuth)*torch.sin(elevation)*w[0][0])-(torch.cos(azimuth)*torch.sin(elevation)*w[0][1]))
       #  jac_h_41 = torch.reshape(torch.stack([jac_h_41_1, jac_h_41_2, jac_h_41_3]), (1, 3))
       #
       #  zeros_block = torch.zeros(1,3) #jack_h_12,jack_h_22,jack_h_32
       #
       #
       #  jac_h_1 = torch.cat((jac_h_11, zeros_block), dim=1)
       #  jac_h_2 = torch.cat((jac_h_21, zeros_block), dim=1)
       #  jac_h_3 = torch.cat((jac_h_31, zeros_block), dim=1)
       #  jac_h_4 =  torch.cat((jac_h_41, jac_h_42), dim=1)
       #  jac_top = torch.cat((jac_h_1, jac_h_2), dim=0)
       #  jac_bottom = torch.cat((jac_h_3, jac_h_4), dim=0)
       #  jac_h =torch.cat((jac_top, jac_bottom), dim=0)

        jac_h =autograd_h(x,tracker_state) #torch.autograd.grad(output, x, create_graph=True)[0]

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
        if delta_x == torch.tensor(0):
            if delta_y > 0:
                azimuth = (torch.pi) / 2
                elevation = torch.atan(delta_z / d)
                # TODO: validate the radian velocity formula
                radian_velocity = torch.dot(los, dv) / d
                doppler_shift = (gamma * radian_velocity) / (2 * lamda)
            else:
                azimuth = (torch.pi) * 1.5
                elevation = torch.atan(delta_z / d)
                # TODO: validate the radian velocity formula
                radian_velocity = torch.dot(los, dv) / d
                doppler_shift = (gamma * radian_velocity) / (2 * lamda)
        else:
            azimuth = torch.atan(delta_y / delta_x)
            elevation = torch.atan(delta_z / d)
            # TODO: validate the radian velocity formula
            radian_velocity = torch.dot(los, dv) / d
            doppler_shift = (gamma * radian_velocity) / (2 * lamda)
        if d == torch.tensor(0):
            azimuth = torch.tensor(0)
            elevation = torch.tensor(0)
            doppler_shift = torch.tensor(0)
            radian_velocity = torch.tensor(0)

        o = torch.stack((gamma_d_2, azimuth, elevation, doppler_shift))
        o = torch.reshape(o[:], (n, 1))
        return o

    # Compute the Jacobian of h with respect to x
    jac_h = torch.autograd.functional.jacobian(h, x).view(n,m)
    return jac_h
###############################################
### process noise Q and observation noise R ###
###############################################
Q_non_diag = False
R_non_diag = False

Q_structure = torch.eye(m)
R_structure = torch.eye(n)

if(Q_non_diag):
    q_d = 1
    q_nd = 1/2
    Q = torch.tensor([[q_d, q_nd, q_nd],[q_nd, q_d, q_nd],[q_nd, q_nd, q_d]])

if(R_non_diag):
    r_d = 1
    r_nd = 1/2
    R = torch.tensor([[r_d, r_nd, r_nd],[r_nd, r_d, r_nd],[r_nd, r_nd, r_d]])

##################################
### Utils for non-linear cases ###
##################################
def getJacobian(x,tracker_state , g):
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
    if g==h:
        _,Jac = g(x , tracker_state, jacobian=True)
    else:
        _, Jac = g(x, jacobian=True)
    return Jac

# T = 10  # number of time steps
#
# x = torch.tensor([[-2.9941e-01],         [ 4.0153e-01],         [ 9.0000e+01],         [-2.9924e-01],         [ 4.0345e-01],         [-2.2596e-05]])
# Trackerstate=torch.tensor([[-5.0000e+01],
#         [-5.0000e+01],
#         [ 8.0000e+01],
#         [-3.0000e-01],
#         [ 4.0000e-01],
#         [ 1.0000e-05]])
#
# x = [1., 2., 3., 4., 5., 6.]
# h_x, J = Auto_h(x, Trackerstate, True)
# manual_h,manual_J = h(x, Trackerstate,True)
# print("h_x: ", h_x)
# print("manual h : ", manual_h)
# print("Jacobian: \n", J)
# print("manual jacobian : \n ", manual_J)
