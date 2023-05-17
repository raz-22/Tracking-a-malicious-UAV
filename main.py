import torch
torch.pi = torch.acos(torch.zeros(1)).item() * 2  # which is 3.1415927410125732
from SysModel import SystemModel
from datetime import datetime
from Parameters import  f, h, m1x_0, m2x_0,m, n, tracker_state, Init_Cov_Matrix
from UAV import *
import math as mt
import numpy as np
from numpy import linspace
from Utils import *
import matplotlib;
#matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits import mplot3d
from EKF import ExtendedKalmanFilter
from MLP import MLP
"seee i did push"
class Environment:
    def __init__(self):
        ###Init UAV Objects ###
        self.tracker = UAV(tracker_state)
        self.target = UAV(m1x_0)
        ### Target Estimated State ###
        self.tgt_est_state = None

        ### Containers to Save The Entire Trajectories ###
        self.tgt_real_traj = torch.empty((0,3,1))
        self.tracker_traj = torch.empty((0,3,1))
        self. tgt_est_traj = torch.empty((0,3,1))

        ### Dynamic Model ###
        self.delta_t = 1
        self.h = h
        self.f = f
        self.Q, self.R = Init_Cov_Matrix()

        # Currently R = self.Q
        self.Dynamic_Model = SystemModel(f=self.f, Q=self.Q, h=self.h, R=self.R,
                                         m=m, n=n, T=1, T_test=1)
        self.Dynamic_Model.InitSequence( self.target.state, m2x_0)  # x0 and P0
        self.Estimator = ExtendedKalmanFilter(self.Dynamic_Model)

    def Update_state(self,est_state = None, target_state=None,tracker_state=None):
        def check_equality(object,instances):
            not_equal = all(elem.any()!=instances[0].any() for elem in instances)
            if not_equal:
                string = "the "+object+" State is not equal throughout all Objects "
                raise ValueError(string)
        if target_state is not None:
            self.target.Update_state(target_state)
            self.tgt_real_traj = torch.cat( [self.tgt_real_traj,target_state[:3, :].unsqueeze(0)], dim = 0)
            check_equality(object = "Target", instances=[target_state,self.target.state,self.Dynamic_Model.x])
        if tracker_state is not None:
            self.tracker.Update_state(tracker_state)
            self.tracker_traj = torch.cat([self.tracker_traj, tracker_state[:3, :].unsqueeze(0)], dim= 0)
            check_equality(object="Tracker", instances=[tracker_state, self.tracker.state])
        if est_state is not None:
            self.tgt_est_state = est_state
            self.tgt_est_traj = torch.cat([self.tgt_est_traj, est_state[:3, :].unsqueeze(0)], dim=0)


    def control_step(self,model = None ,module = "mlp"):
        """
        The Tracking UAV control module.
        Decides the next velocity, azimuth & elevation based on Target State Estimation .
        Can be constant decision or Multi Layer Perceptron NN

        :param param1: NN model for the case of MLP control module
        :type param1: class item

        :param param2: Flag Deciding what type of control module to use (constant/NN)
        :type param2: String

        :return: Velocity ,Azimuth & Elevation.
        :rtype: tensor (,)
        """
        #####################################
        ##  Use selected Decision Module   ##
        #####################################
        if module == "mlp":
            navigation_decision = model.forward(self.target.state,self.tracker.state)
            return navigation_decision[0],navigation_decision[1],navigation_decision[2]
        elif module =="fixed":
            #####################################
            ####  Calculate Gradient on Cost  ###
            #####################################
            #####################################
            return torch.tensor(100), torch.tensor(50), torch.tensor(50)
        elif module =="random":
            #####################################
            ####  Calculate Gradient on Cost  ###
            #####################################
            #####################################
            # random (0.5<v<>1), (0<azimuth<50) , (0<elevation<50)
            #return torch.tensor(100), torch.tensor(50), torch.tensor(50)
            pass
    def step(self, model = None, mode = "test", step=0):
        """
        Executing one time step of our Simulator

        :param param1: NN model for the case of MLP control module
        :type param1: class item

        :param param2: Flag Deciding what type of step to use (//)
        :type param2: String

        :return: .
        :rtype: .
        """
        ##  Step & Observation ##
        self.Dynamic_Model.GenerateStep(tracker_state=self.tracker.state)
        self.Update_state(target_state=self.Dynamic_Model.x)

        ### State Estimation ###
        self.Estimator.Update(y=self.Dynamic_Model.y,
                              tracker_state=self.tracker.state, step=step)
        if mode == "test":
            ###### Control Law ######
            v, heading, tilt = self.control_step(module = "fixed")

            ###### Update State ######
            self.Update_state(est_state = self.Estimator.m1x_posterior,
                              tracker_state = (self.tracker.next_position( v, heading, tilt)))
        elif mode == "train single step":
            return {"m2x_prior": self.Estimator.m2x_prior, "jac_H": self.Estimator.batched_H, "R": self.R}
        elif mode == "train_sequential":
            ###### Control Law ######
            v, heading, tilt = self.control_step(model= model, module="mlp")

            ###### Update State ######
            self.Update_state(est_state=self.Estimator.m1x_posterior
                              , tracker_state=self.tracker.next_position(v, heading, tilt))
            return self.Estimator.m2x_posterior, self.Estimator.m2x_prior,self.Estimator.batched_H, self.Estimator.KG

def train(env,model , num_steps=10000):
    # Define the optimizer and compile the model
    optimizer = torch.optim.Adam(model.parameters())
    running_loss = 0.0
    custom_loss = InfromationTheoreticCost(weight=1)
    for step in range(num_steps):
        model.train()
        args = env.step(mode="train single step")
        loss = custom_loss(args, mode="single")
        running_loss += loss.item()
        optimizer.zero_grad()
        loss.backward(retain_graph = True)
        optimizer.step()
        ###### Control Law ######
        v, heading, tilt = env.control_step(model =model,module = "mlp")

        ###### Update State ######
        if (abs(v) or abs(heading) or abs(tilt)) >50 :
            raise ValueError("Single step trainf: Control decision is out of limits")
        env.Update_state(est_state = env.Estimator.m1x_posterior,tracker_state = env.tracker.next_position( v, heading, tilt))
        if step%10 ==0:
            # Print the average loss for the epoch
            print("step %d loss: %.3f" % (step + 1, running_loss / (step+1)))

def constant_ctrl_simulation(env, num_steps=99):
    # Set up the 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.set_title("3D Live Simulation")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    for i in range(num_steps):
        # Clear the plot at the start of each step
        ax.clear()
        ax.set_title("3D Live Simulation")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

        env.step()
        target_real_traj = env.tgt_real_traj[i].squeeze().tolist()
        target_est_traj = env.tgt_est_traj[i].squeeze().tolist()
        tracker_traj = env.tracker_traj[i].squeeze().tolist()
        target_delta_traj = torch.norm(env.tgt_real_traj[i].squeeze()-env.tgt_est_traj[i].squeeze())

        if target_delta_traj>10:
            print("large delta")
        if torch.isnan(target_delta_traj):
            print("nan")

        # Update the target position on the 3D plot
        ax.scatter(*target_real_traj, c="r", marker="o", label="Target Real")
        # Update the target position on the 3D plot
        ax.scatter(*target_est_traj, c="g", marker="*", label="Target Estimated")
        # Update the tracker position on the 3D plot
        ax.scatter(*tracker_traj, c="b", marker="^", label="Tracker")

        # Update the plot every step and pause briefly
        ax.legend()
        plt.pause(0.01)

        mse = estimation_mse_loss(env.tgt_est_traj, env.tgt_real_traj)
        print("Mean Squared Error between est_state and real_state: ",mse)

#
# if __name__ == '__main__':
#     print("Pipeline Start")
#
#     today = datetime.today()
#     now = datetime.now()
#     strToday = today.strftime("%m.%d.%y")
#     strNow = now.strftime("%H:%M:%S")
#     strTime = strToday + "_" + strNow
#     print("Current Time =", strTime)
#
#
#     model = MLP()
#     model.initialize_weights()
#     train_sequential(model, num_steps=100)
#     #env.train(model)
#     #env.generate_simulation(num_steps=1000)
#
#     print("Pipeline Ends")
#     today = datetime.today()
#     now = datetime.now()
#     strToday = today.strftime("%m.%d.%y")
#     strNow = now.strftime("%H:%M:%S")
#     strTime = strToday + "_" + strNow
#     print("Current Time =", strTime)
