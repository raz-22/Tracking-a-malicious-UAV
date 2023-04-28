import torch
torch.pi = torch.acos(torch.zeros(1)).item() * 2  # which is 3.1415927410125732
from SysModel import SystemModel
from datetime import datetime
from parameters import  f, h, m1x_0, m2x_0, n, target_state, tracker_state
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
        self.tracker_state =tracker_state
        self.target_state = target_state
        self.tracker = Tracker(tracker_state)
        self.target = Target(target_state)
        self.delta_t = 1
        self.h = h
        self.f = f
        self.Q, self.R = self.Init_Cov_Matrix()
        self.tracker_traj =[self.tracker_state[:3, :]]
        # TODO: go over all init parameters

        # Currently R = self.Q
        self.Dynamic_Model = SystemModel(f=self.f, Q=self.Q, h=self.h, R=self.R,
                                         m=6, n=4, T=1, T_test=1, prior_Q=None, prior_Sigma=None, prior_S=None)
        self.Dynamic_Model.InitSequence(m1x_0, m2x_0)  # x0 and P0
        self.Estimator = ExtendedKalmanFilter(self.Dynamic_Model, 'none')

    def Update_state(self,target_state,tracker_state):
        self.tracker_state = tracker_state
        self.target_state = target_state
        self.tracker.Update_state(tracker_state)
        self.target.Update_state(target_state)

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
        #TODO: decide with nir on values
        diagonal = [1e-5, 1e-5, 1e-6, 1e-6]
        diagonal_matrix = torch.diag(torch.tensor(diagonal))
        R = torch.eye(n)*diagonal_matrix
        return Q, R

    def calculate_cost(self, matrix):
        """Calculates -ln(det(matrix))"""
        det = torch.det(matrix)
        cost = -torch.log(det)
        return cost
    def information_theoretic_cost(self):
        cost = torch.inverse(self.m2x_prior) + torch.matmul(torch.transpose(self.jac_H, 0, 1), torch.matmul(torch.inverse(self.R), self.jac_H))
        return cost
    def control_step(self):
        ###############################################
        ####  Calculate Information Theoretic Cost  ###
        ###############################################
        inf_theor_cost = self.information_theoretic_cost()
        #########################
        ####  Calculate Cost  ###
        #########################
        cost = self.calculate_cost(inf_theor_cost)
        #########################
        ####  Calculate Gradient on Cost  ###
        #########################



    def step(self):
        #########################
        ##  Step & Observation ##
        #########################
        self.Dynamic_Model.UpdateCovariance_Matrix(self.Q,self.R)
        self.Dynamic_Model.GenerateStep(Q_gen=self.Q, R_gen=self.R,tracker_state = self.tracker.state) #updates Dynamic_Model.x,.y,.x_prev
        #########################
        ### State Estimation ###
        #########################

        self.m1x_posterior, self.m2x_posterior, self.m2x_prior, self.m2y, self.jac_H = self.Estimator.Update(self.Dynamic_Model.y, self.Dynamic_Model.m1x_0, self.Dynamic_Model.m2x_0, tracker_state = self.tracker.state)

        self.Dynamic_Model.m1x_0 = self.m1x_posterior
        self.Dynamic_Model.m2x_0 = self.m2x_posterior
        #########################
        ###### Control Law ######
        #########################
        # v, heading, tilt = self.control_step()

        v, heading, tilt = torch.tensor(100), torch.tensor(50), torch.tensor(50)

        tracker_state = self.tracker.next_position( v, heading, tilt)
        self.tracker_traj.append(tracker_state[:3, :])
#        self.Dynamic_Model.y.backward()
#        print(self.target_state.grad)
        self.Update_state(self.m1x_posterior,tracker_state)
       # self.control_step()


    #Fixme: becomes nan after 192~ steps
    

    def generate_simulation(self, num_steps=50):
        coords = []
        est_state = []
        for i in range(num_steps):
            self.step()
            self.target.current_position = torch.reshape(self.target.current_position, (3, 1))
            coords.append(self.target.current_position[:, 0])
            est_state.append(torch.reshape(self.target_state[:, 0], (6,1)))
            # for v in self.target.current_position[:, 0]:
            #     # if torch.isnan(v):
            #     #     print('nan')
        
            print(f"Step {i + 1}: {self.target.current_position[:, 0]}")  # Print the location at each step
        coords = np.stack(coords)
        real_state = np.stack(self.Dynamic_Model.real_traj[:])
        real_traj = real_state[:,:3,:]
        real_traj = real_traj[:,:,0]
        est_state = np.stack(est_state[:])
        tracker_traj=np.stack(self.tracker_traj)[:50,:,0]
        print("Real Trajectory:")
        for i, pos in enumerate(real_traj):
            print(f"Step {i + 1}: {pos[:]}")

        mse = calculate_mse(est_state, real_state)
        print(f"Mean Squared Error between est_state and real_state: {mse}") 

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        ##################################
        ###### Estimated Trajectory ######
        ##################################
        # Create tail
        tail, = ax.plot([], [], [], c='r')
        # Create newest dot
        dot, = ax.plot([], [], [], 'o', c='b')
        ##################################
        ######## Real Trajectory #########
        ##################################
        real_tale = ax.plot([], [], [], c='m', linestyle='--', linewidth=2)[0]
        real_dot = ax.plot([], [], [], 's', c='g', markersize=10)[0]
        ##################################
        ######## Tracker Trajectory ######
        ##################################
        tracker_tale = ax.plot([], [], [], c='purple', linestyle=':', linewidth=2)[0]
        tracker_dot = ax.plot([], [], [], 'o', c='orange', markersize=10, markeredgecolor='black', markeredgewidth=1.5)[0]

        # Add the labels to the dots
        dot.set_label('Estimated UAV')
        real_dot.set_label('Real UAV')
        tracker_dot.set_label('Tracker UAV')

        # Add the labels to the plot
        ax.legend()

        # Set axis limits based on min/max values in coords
        x_min, x_max = coords[:, 0].min(), coords[:, 0].max()
        y_min, y_max = coords[:, 1].min(), coords[:, 1].max()
        z_min, z_max = coords[:, 2].min(), coords[:, 2].max()

        if not (np.isnan(x_min) or np.isinf(x_min) or np.isnan(x_max) or np.isinf(x_max)):
            ax.set_xlim3d(x_min, x_max)
        if not (np.isnan(y_min) or np.isinf(y_min) or np.isnan(y_max) or np.isinf(y_max)):
            ax.set_ylim3d(y_min, y_max)
        if not (np.isnan(z_min) or np.isinf(z_min) or np.isnan(z_max) or np.isinf(z_max)):
            ax.set_zlim3d(z_min, z_max)


        # Set animation function
        ani = animation.FuncAnimation(fig, update_graph,len(coords), fargs=(coords, real_traj, tracker_traj,  tail,dot, real_tale, real_dot, tracker_tale, tracker_dot), interval=50)#, blit=False)

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
    env.generate_simulation()
    print("Pipeline Ends")
    today = datetime.today()
    now = datetime.now()
    strToday = today.strftime("%m.%d.%y")
    strNow = now.strftime("%H:%M:%S")
    strTime = strToday + "_" + strNow
    print("Current Time =", strTime)



