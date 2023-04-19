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
m1x_0 = torch.ones(m, 1) 
m2x_0 = 0 * 0 * torch.eye(m)

### Decimation
delta_t_gen =  1e-5
delta_t = 1 # 0.02
ratio = delta_t_gen/delta_t

###Observation Function Parameters
gamma = 4
lamda = 3.8961*(1e-3)

### Taylor expansion order
J = 5 
J_mod = 2
"""
### Angle of rotation in the 3 axes
roll_deg = yaw_deg = pitch_deg = 1

roll = roll_deg * (math.pi/180)
yaw = yaw_deg * (math.pi/180)
pitch = pitch_deg * (math.pi/180)

RX = torch.tensor([
                [1, 0, 0],
                [0, math.cos(roll), -math.sin(roll)],
                [0, math.sin(roll), math.cos(roll)]])
RY = torch.tensor([
                [math.cos(pitch), 0, math.sin(pitch)],
                [0, 1, 0],
                [-math.sin(pitch), 0, math.cos(pitch)]])
RZ = torch.tensor([
                [math.cos(yaw), -math.sin(yaw), 0],
                [math.sin(yaw), math.cos(yaw), 0],
                [0, 0, 1]])

RotMatrix = torch.mm(torch.mm(RZ, RY), RX)

### Auxiliar MultiDimensional Tensor B and C (they make A --> Differential equation matrix)
C = torch.tensor([[-10, 10,    0],
                  [ 28, -1,    0],
                  [  0,  0, -8/3]]).float()
"""
######################################################
### State evolution function f for  UAV usecase ###
######################################################
#TODO: understand F_GEN
"""
### f_gen is for dataset generation
def f_gen(x, jacobian=False):
    BX = torch.zeros([x.shape[0],m,m]).float() #[batch_size, m, m]
    BX[:,1,0] = torch.squeeze(-x[:,2,:]) 
    BX[:,2,0] = torch.squeeze(x[:,1,:])
    
    Const = C
    A = torch.add(BX, Const)  
    # Taylor Expansion for F    
    F = torch.eye(m)
    F = F.reshape((1, m, m)).repeat(x.shape[0], 1, 1) # [batch_size, m, m] identity matrix
    for j in range(1,J+1):
        F_add = (torch.matrix_power(A*delta_t_gen, j)/math.factorial(j))
        F = torch.add(F, F_add)
    if jacobian:
        return torch.bmm(F, x), F
    else:
        return torch.bmm(F, x)
"""
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

def h(x, target_state,tracker_state,jacobian=False):
    los = target_state[0] - tracker_state[0]
    dv = target_state[1] - tracker_state[1]
    delta_x = los[0]
    delta_y = los[1]
    delta_z = los[2]

    # h(s) observstion function measurement equations
    d = torch.norm(los)
    gamma_d_2 = (gamma / 2) * (d)
    azimuth = torch.atan(delta_y / delta_x)
    elevation = torch.atan(delta_z / d)
    # TODO: validate the radian velocity formula
    radian_velocity = torch.dot(tracker_state[1].transpose(), los) / d

    doppler_shift = (4 * radian_velocity) / (2 * lamda)
    o = torch.stack((gamma_d_2, azimuth, elevation, doppler_shift))

    if jacobian:
        arg_1 = azimuth
        arg_2 = elevation
        a = torch.transpose(torch.stack([(torch.cos(arg_1)*torch.sin(arg_2)),(torch.sin(arg_1)*torch.sin(arg_2)),torch.cos(arg_2)]))
        #todo: breakpoint
        jac_h_11 = (gamma/2)*a
        jac_h_42 = (gamma / (2 * lamda)) * a

        arg_1 = (azimuth+(torch.pi/2))
        arg_2 = (torch.pi/2)
        a = torch.transpose(torch.stack( [(torch.cos(arg_1) * torch.sin(arg_2)), (torch.sin(arg_1) * torch.sin(arg_2)),torch.cos(arg_2)]))
        jac_h_21 = a/(d*torch.sin(elevation))

        arg_1 = azimuth
        arg_2 = elevation+(torch.pi / 2)
        a = torch.transpose(torch.stack([(torch.cos(arg_1) * torch.sin(arg_2)), (torch.sin(arg_1) * torch.sin(arg_2)), torch.cos(arg_2)]))
        jac_h_31 = a/d

        w = torch.cross(los,dv)/(d**2)
        jac_h_41_1 = (gamma/(2*lamda))*((torch.cos(elevation)*w[1])-(torch.sin(azimuth)*torch.sin(elevation)))
        jac_h_41_2 = (gamma / (2 * lamda)) * ((torch.cos(azimuth)*torch.sin(elevation)*w[2])-(torch.cos(elevation)*w[0]))
        jac_h_41_3 = (gamma / (2 * lamda)) * ((torch.sin(azimuth)*torch.sin(elevation)*w[0])-(torch.cos(azimuth)*torch.sin(elevation)*w[1]))
        jac_h_41 = torch.stack([jac_h_41_1, jac_h_41_2, jac_h_41_3])

        zeros_block = torch.zeros(3,2) #jack_h_12,jack_h_22,jack_h_32


        jac_h_1 = torch.cat((jac_h_11, zeros_block), dim=1)
        jac_h_2 = torch.cat((jac_h_21, zeros_block), dim=1)
        jac_h_3 = torch.cat((jac_h_31, zeros_block), dim=1)
        jac_h_4 =  torch.cat((jac_h_41, jac_h_42), dim=1)
        jac_top = torch.cat((jac_h_1, jac_h_2), dim=0)
        jac_bottom = torch.cat((jac_h_3, jac_h_4), dim=0)
        jac_h =torch.cat((jac_top, jac_bottom), dim=0)

        return o, jac_h
    else:
        return o


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
def getJacobian(x, g):
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
    _,Jac = g(x, jacobian=True)
    return Jac

T = 10  # number of time steps  

