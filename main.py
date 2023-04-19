import torch
torch.pi = torch.acos(torch.zeros(1)).item() * 2  # which is 3.1415927410125732
from SysModel import SystemModel
from datetime import datetime
from parameters import  f, h, m1x_0, m2x_0, m, n, Q_structure, R_structure
from UAV import *
import math as mt
import numpy as np
from numpy import linspace
from utils import *
import matplotlib;
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits import mplot3d
from EKF import ExtendedKalmanFilter

class Environment:
    def __init__(self):
        self.tracker = Tracker()
        self.target = Target()
        self.delta_t = 1
        self.h = h
        self.Q = self.Init_Cov_Matrix()
        self.R = self.Q
        # TODO: go over all init parameters
        self.sys_model = SystemModel(f, self.Q, h, self.Q, 1, 1, m, n)  # parameters for GT
        self.sys_model.InitSequence(m1x_0, m2x_0)  # x0 and P0


        # Currently R = self.Q
        self.Dynamic_Model = SystemModel(f=f, Q=self.Q, h=self.h, R=self.Q,
                                 m=6, n=4, T=1, T_test=1, prior_Q=None, prior_Sigma=None, prior_S=None)
        self.Estimator = ExtendedKalmanFilter(self.Dynamic_Model, "cuda")
        self.target_state = self.initialize_target_state()
        self.tracker_state = self.initialize_tracker_state()

    def Init_Cov_Matrix(self):
        # Calculating the noise covariance matrix , constants are from the use case in the original paper
        diagonal = [1e-5, 1e-5, 1e-6]
        diagonal_matrix = torch.diag(torch.tensor(diagonal))
        delta_t = 0.1
        A = ((delta_t ** 3) / 3) * diagonal_matrix
        B = ((delta_t ** 2) / 2) * diagonal_matrix
        C = ((delta_t ** 2) / 2) * diagonal_matrix
        D = delta_t * diagonal_matrix
        top = torch.cat((A, B), dim=1)
        bottom = torch.cat((C, D), dim=1)
        Q = torch.cat((top, bottom), dim=0)
        return Q

    def step(self):
        self.Dynamic_Model.UpdateCovariance_Matrix(self.Q,self.R)
        self.Dynamic_Model.GenerateStep(Q_gen=self.Q, R_gen=self.R, target_state=self.target_state, tracker_state=self.tracker_state) #updates Dynamic_Model.x,.y,.x_prev

        #self.Estimator.step(self.Dynamic_Model.y)

        # self.control_step()

        self.tracker.next_position()

    def initialize_target_state(self):
            # Define the initial target state, depending on your specific problem
            initial_target_state =  torch.zeros(size=[m, 1]) # ULI: INIT is not correct 
            return initial_target_state

    def initialize_tracker_state(self):
        # Define the initial tracker state, depending on your specific problem
        initial_tracker_state = torch.zeros(size=[m, 1]) # ULI: INIT is not correct 
        return initial_tracker_state
    
    def generate_simulation(self, num_steps=1000):
        # Create a list to store the last 10 positions
        coords = []

        for i in range(num_steps):
            env.step()
            coords.append(env.target.current_position[:, 0])
        coords = np.array(coords)
        # coords = np.random.rand(100, 3)
        # Create a figure and 3D axes
        fig = plt.figure()
        ax = plt.axes(111, projection='3d')
        # Create tail

        tail, = ax.plot([], [], [], c='r')
        # Create newest dot
        dot, = ax.plot([], [], [], 'o', c='b')

        # Set axis limits based on min/max values in coords
        ax.set_xlim3d(coords[:, 0].min(), coords[:, 0].max())
        ax.set_ylim3d(coords[:, 1].min(), coords[:, 1].max())
        ax.set_zlim3d(coords[:, 2].min(), coords[:, 2].max())

        # Set animation function
        ani = animation.FuncAnimation(fig, update_graph, len(coords), fargs=(coords, tail, dot),
                                      interval=50)  # , tail), interval=50)

        # Show plot
        plt.show()


if __name__ == '__main__':
    print("Pipeline Start")
    ################
    ### Get Time ###
    ################
    today = datetime.today()
    now = datetime.now()
    strToday = today.strftime("%m.%d.%y")
    strNow = now.strftime("%H:%M:%S")
    strTime = strToday + "_" + strNow
    print("Current Time =", strTime)

    env = Environment()
    env.Dynamic_Model.InitSequence(m1x_0, m2x_0)
    env.generate_simulation()



