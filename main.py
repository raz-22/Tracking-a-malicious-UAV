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
from mlp import MLP
class Environment:
    def __init__(self):
        ##Changing in states are through the Objects tracker/target
        if m1x_0.any() != target_state.any():
            raise ValueError("m1x_0 and initial tracker state must be equal")
        self.tracker = Tracker(tracker_state)
        self.target = Target(target_state)
        self.tgt_est_state = None
        self.m2x_posterior = m2x_0
        self.tgt_real_state = self.target.state
        self.tracker_state = self.tracker.state

        #self.tracker_state =tracker_state
        #self.target_state = target_state
        #self.tracker = Tracker(tracker_state)
        #self.target = Target(target_state)
        self.tgt_real_traj = torch.empty((0,3,1)) #torch.cat([self.tgt_real_state[:3]], dim = 0).reshape(0,3,1)
        self.tracker_traj = torch.empty((0,3,1)) #torch.cat([self.tracker_state[:3]], dim =0).reshape(0,3,1)
        self. tgt_est_traj = torch.empty((0,3,1))
        self.m1x_posterior = m1x_0
        #self.m2x_posterior = m2x_0

        self.delta_t = 1
        self.h = h
        self.f = f
        self.Q, self.R = self.Init_Cov_Matrix()

        # TODO: go over all init parameters

        # Currently R = self.Q
        self.Dynamic_Model = SystemModel(f=self.f, Q=self.Q, h=self.h, R=self.R,
                                         m=6, n=4, T=1, T_test=1, prior_Q=None, prior_Sigma=None, prior_S=None)
        self.Dynamic_Model.InitSequence(target_state, m2x_0)  # x0 and P0
        self.Estimator = ExtendedKalmanFilter(self.Dynamic_Model, 'none')
        self.model = MLP()
        self.model.initialize_weights()
    def Update_state(self,est_state = None, target_state=None,tracker_state=None):
        if target_state is not None:
            self.tgt_real_state = target_state
            self.target.Update_state(target_state)
            self.tgt_real_traj = torch.cat( [self.tgt_real_traj,target_state[:3, :].unsqueeze(0)], dim = 0)
            #self.tgt_real_traj.append(target_state[:3, :])
        if tracker_state is not None:
            self.tracker_state = tracker_state
            self.tracker.Update_state(tracker_state)
            self.tracker_traj = torch.cat([self.tracker_traj, tracker_state[:3, :].unsqueeze(0)], dim= 0)
            #self.tracker_traj.append(tracker_state[:3, :])
        if est_state is not None:
            tgt_est_state = est_state
            self.tgt_est_traj = torch.cat([self.tgt_est_traj, est_state[:3, :].unsqueeze(0)], dim=0)
            #self.tgt_est_traj.append(est_state[:3, :])
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
    def control_step(self, module = "mlp"):
        ###############################################
        ####  Calculate Information Theoretic Cost  ###
        ###############################################
        inf_theor_cost = self.information_theoretic_cost()
        #########################
        ####  Calculate Cost  ###
        #########################
        cost = self.calculate_cost(inf_theor_cost)
        #####################################
        ####  Calculate Gradient on Cost  ###
        #####################################
        #####################################
        ##  Use selected Decision Module   ##
        #####################################
        if module == "mlp":
            input = torch.cat((self.target.state.squeeze(),self.tracker.state.squeeze()),dim = 0)
            navigation_decision = self.model.forward(input)
            return navigation_decision[0],navigation_decision[1],navigation_decision[2], cost
        elif module =="rnn":
            pass
        elif module == "analityc":
            pass
        elif module =="fixed":
            return torch.tensor(100), torch.tensor(50), torch.tensor(50), cost

    def step(self, mode = "test"):
        if mode == "test":
            #########################
            ##  Step & Observation ##
            #########################
            self.Dynamic_Model.UpdateCovariance_Matrix(self.Q,self.R)
            tgt_state = self.Dynamic_Model.GenerateStep(Q_gen=self.Q, R_gen=self.R,tracker_state = self.tracker.state) # updates Dynamic_Model.x,.y,.x_prev
            self.Update_state(target_state = tgt_state)
            ########################
            ### State Estimation ###
            ########################
            self.m1x_posterior, self.m2x_posterior, self.m2x_prior, self.m2y, self.jac_H = self.Estimator.Update(self.Dynamic_Model.y, self.m1x_posterior, self.m2x_posterior, tracker_state = self.tracker.state)
            #self.Dynamic_Model.m1x_0 = self.m1x_posterior
            #self.Dynamic_Model.m2x_0 = self.m2x_posterior
            #########################
            ###### Control Law ######
            #########################
            v, heading, tilt, cost = self.control_step(module = "fixed")
            #########################
            ###### Update Stat ######
            #########################
            tracker_state = self.tracker.next_position( v, heading, tilt)
            self.Update_state(est_state = self.m1x_posterior,tracker_state = tracker_state)
        elif mode == "train":
            #########################
            ### State Estimation ###
            #########################
            self.m1x_posterior, self.m2x_posterior, self.m2x_prior, self.m2y, self.jac_H = self.Estimator.Update(
                self.Dynamic_Model.y, self.Dynamic_Model.m1x_0, self.Dynamic_Model.m2x_0,
                tracker_state=self.tracker.state)
            self.Dynamic_Model.m1x_0 = self.m1x_posterior
            self.Dynamic_Model.m2x_0 = self.m2x_posterior
            #########################
            ###### Control Law ######
            #########################
            v, heading, tilt, cost = self.control_step()
            #########################
            ###### Update Stat ######
            #########################
            tracker_state = self.tracker.next_position(v, heading, tilt)
            self.tracker_traj.append(tracker_state[:3, :])
            self.Update_state(self.m1x_posterior, tracker_state)

    #Fixme: becomes nan after 192~ steps

    def train(self, module="mlp", num_steps = 99):
        if module == "mlp":
            pass
    
    def generate_simulation(self, num_steps= 99):
        # Set up the 3D plot
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.set_title("3D Live Simulation")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

        with open("output.txt", "w") as file:
            for i in range(num_steps):
                self.step()
                target_pos = self.target.current_position.tolist()
                print(f"Target Step {i + 1}: {target_pos}", file=file)

                # Update the target position on the 3D plot
                ax.scatter(*target_pos, c="r", marker="o", label="Target" if i == 0 else "")
                
                if i < len(self.tgt_real_traj):
                    tracker_pos = self.tgt_real_traj[i].squeeze().tolist()
                    print(f"Tracker Step {i + 1}: {tracker_pos}", file=file)

                    # Update the tracker position on the 3D plot
                    ax.scatter(*tracker_pos, c="b", marker="^", label="Tracker" if i == 0 else "")

                # Update the plot every step and pause briefly
                if i == 0:
                    ax.legend()
                plt.pause(0.01)

            mse = calculate_loss(self.tgt_est_traj, self.tgt_real_traj)
            print(f"Mean Squared Error between est_state and real_state: {mse}", file=file)

        # Keep the plot open after the simulation
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



