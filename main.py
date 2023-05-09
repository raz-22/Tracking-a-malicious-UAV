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
            self.tgt_est_state = est_state
            self.tgt_est_traj = torch.cat([self.tgt_est_traj, est_state[:3, :].unsqueeze(0)], dim=0)
            #self.tgt_est_traj.append(est_state[:3, :])
    def Init_Cov_Matrix(self):
        # Calculating the noise covariance matrix , constants are from the use case in the original paper
        diagonal = [1e-5, 1e-5, 1e-6]
        diagonal_matrix = torch.diag(torch.tensor(diagonal))
        delta_t = 1
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

    def control_step(self,model = None ,module = "mlp"):
        #####################################
        ##  Use selected Decision Module   ##
        #####################################
        if module == "mlp":
            navigation_decision = model.forward(self.target.state,self.tracker.state)
            return navigation_decision[0],navigation_decision[1],navigation_decision[2]
        elif module =="rnn":
            pass
        elif module == "analityc":
            pass
        elif module =="fixed":
            #####################################
            ####  Calculate Gradient on Cost  ###
            #####################################
            #####################################
            return torch.tensor(100), torch.tensor(50), torch.tensor(50)

    def step(self,file = None, model = None, mode = "test"):
        if mode == "test":
            #########################
            ##  Step & Observation ##
            #########################

            # print(f"Updatind cov matrix and generating step :", file=file)

            self.Dynamic_Model.UpdateCovariance_Matrix(self.Q,self.R)
            tgt_state = self.Dynamic_Model.GenerateStep(Q_gen=self.Q, R_gen=self.R,tracker_state = self.tracker.state) # updates Dynamic_Model.x,.y,.x_prev

            # print(f"step generated. parameters :", file=file)
            # print(f"Dynamic_Model.x :{self.Dynamic_Model.x}", file=file)
            # print(f"Dynamic_Model.x_prev :{self.Dynamic_Model.x_prev}", file=file)
            # print(f"Dynamic_Model.y :{self.Dynamic_Model.y}", file=file)
            # print(f"Dynamic_Model.m1x_0 :{self.Dynamic_Model.m1x_0}", file=file)
            # print(f"Dynamic_Model.m2x_0 :{self.Dynamic_Model.m2x_0}", file=file)
            # print(f"Updatind states :", file=file)

            self.Update_state(target_state = tgt_state)

            # print(f"self.tgt_real_traj :{self.tgt_real_traj[-1]}", file=file)


            ########################
            ### State Estimation ###
            ########################

            # print(f"starting EKF : inputs:", file=file)
            # print(f"m1x_posterior : {self.m1x_posterior}", file=file)
            # print(f"actual x : {self.Dynamic_Model.x} ", file=file)
            # print(f"m2x_posterior : {self.m2x_posterior}", file=file)
            # print(f"tracker_state : {self.tracker.state}", file=file)
            # print(f"self.Dynamic_Model.y : {self.Dynamic_Model.y}", file=file)

            self.m1x_posterior, self.m2x_posterior, self.m2x_prior, self.m2y, self.jac_H, self.KG = self.Estimator.Update(y = self.Dynamic_Model.y, tracker_state = self.tracker.state, file =file ,x_temp= self.Dynamic_Model.x)# self.m1x_posterior, self.m2x_posterior, tracker_state = self.tracker.state, file =file ,x_temp= self.Dynamic_Model.x)

            # print(f" EKF : Outputs:", file=file)
            # print(f"m1x_posterior : {self.m1x_posterior}", file=file)
            # print(f"actual x : {self.Dynamic_Model.x} ", file=file)
            # print(f"m2x_posterior : {self.m2x_posterior}", file=file)
            # print(f"m2x_prior : {self.m2x_prior}", file=file)
            # print(f"self.m2y : {self.m2y}", file=file)
            # print(f"self.jac_H : {self.jac_H}", file=file)

            #########################
            ###### Control Law ######
            #########################

            # print(f" control law", file=file)

            v, heading, tilt = self.control_step(module = "fixed")
            #########################
            ###### Update Stat ######
            #########################
            tracker_state = self.tracker.next_position( v, heading, tilt)

            # print(f"Updatind states :", file=file)

            self.Update_state(est_state = self.m1x_posterior,tracker_state = tracker_state)

            # print(f"self.tgt_real_traj :{self.tgt_real_traj[-1]}", file=file)
            # print(f"self.tgt_est_traj :{self.tgt_est_traj[-1]}", file=file)
            # print(f"actual x : {self.Dynamic_Model.x} ", file=file)
            # print(f"self.tracker_traj :{self.tracker_traj[-1]}", file=file)

        elif mode == "train":
            #########################
            ##  Step & Observation ##
            #########################
            self.Dynamic_Model.UpdateCovariance_Matrix(self.Q,self.R)
            tgt_state = self.Dynamic_Model.GenerateStep(Q_gen=self.Q, R_gen=self.R,tracker_state = self.tracker.state) # updates Dynamic_Model.x,.y,.x_prev
            self.Update_state(target_state = tgt_state)
            ########################
            ### State Estimation ###
            ########################
            self.m1x_posterior, self.m2x_posterior, self.m2x_prior, self.m2y, self.jac_H, self.KG = self.Estimator.Update(y = self.Dynamic_Model.y, tracker_state = self.tracker.state, file =file ,x_temp= self.Dynamic_Model.x)# self.m1x_posterior, self.m2x_posterior, tracker_state = self.tracker.state, file =file ,x_temp= self.Dynamic_Model.x)
        elif mode == "train_sequential":
            #########################
            ##  Step & Observation ##
            #########################
            self.Dynamic_Model.UpdateCovariance_Matrix(self.Q, self.R)
            tgt_state = self.Dynamic_Model.GenerateStep(Q_gen=self.Q, R_gen=self.R,
                                                        tracker_state=self.tracker.state)  # updates Dynamic_Model.x,.y,.x_prev
            self.Update_state(target_state=tgt_state)
            ########################
            ### State Estimation ###
            ########################
            self.m1x_posterior, self.m2x_posterior, self.m2x_prior, self.m2y, self.jac_H, self.KG = self.Estimator.Update(
                y=self.Dynamic_Model.y, tracker_state=self.tracker.state, file=file,
                x_temp=self.Dynamic_Model.x)  # self.m1x_posterior, self.m2x_posterior, tracker_state = self.tracker.state, file =file ,x_temp= self.Dynamic_Model.x)
            #########################
            ###### Control Law ######
            #########################
            v, heading, tilt = self.control_step(model= model, module="mlp")
            #########################
            ###### Update Stat ######
            #########################
            tracker_state = self.tracker.next_position(v, heading, tilt)
            self.Update_state(est_state=self.m1x_posterior, tracker_state=tracker_state)
            return  self.m2x_posterior, self.m2x_prior,self.jac_H, self.KG
    #Fixme: becomes nan after 192~ steps
    def train_sequential(self, model, num_steps=10):
        # Define the optimizer and compile the model
        optimizer = torch.optim.Adam(model.parameters())
        custom_loss = InfromationTheoreticCost(weight=1)
        running_loss = 0.0
        #FIXME: sequential loss is only for test purposes
        args, sequential_loss = generate_traj(self,model = model, num_steps= num_steps,mode = "sequential")
        loss = custom_loss(args,mode = "sequential")
        if (loss - sequential_loss)>1e-1:
             print("error in sequential loss calculations")
        print("trajectory average loss is "+(str(loss.item())))
        running_loss+=loss
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()

    def information_theoretic_cost(self,args, mode="single"):
        if mode == "single":
            inf_theor_cost = (torch.inverse(self.m2x_prior) + torch.matmul(torch.transpose(self.jac_H, 0, 1),
                                                                              torch.matmul(torch.inverse(self.R),
                                                                                           self.jac_H)))[:3, :3]
            """Calculates -ln(det(matrix))"""
            det = torch.det(inf_theor_cost)
            if det < 0:
                print("determinant is negative")
            ln_det_cost = -torch.log(det)
            return ln_det_cost
    def train(self,model , num_steps=10000):
        # Define the optimizer and compile the model
        optimizer = torch.optim.Adam(model.parameters())
        running_loss = 0.0
        for step in range(num_steps):
            model.train()
            self.step(mode="train")
            args = {"m2x_prior":self.m2x_prior,"jac_H":self.jac_H,"R":self.R}
            loss = self.information_theoretic_cost(args, mode= "single")
            running_loss += loss.item()
            optimizer.zero_grad()
            loss.backward(retain_graph = True)
            optimizer.step()
            #########################
            ###### Control Law ######
            #########################
            v, heading, tilt = self.control_step(model =model,module = "mlp")
            #########################
            ###### Update Stat ######
            #########################
            if (abs(v) or abs(heading) or abs(tilt)) >50 :
                print("out of limits")

            tracker_state = self.tracker.next_position( v, heading, tilt)
            self.Update_state(est_state = self.m1x_posterior,tracker_state = tracker_state)
            if step%10 ==0:
                # Print the average loss for the epoch
                print("step %d loss: %.3f" % (step + 1, running_loss / (step+1)))
    
    def generate_simulation(self, num_steps= 99):
        # Set up the 3D plot
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.set_title("3D Live Simulation")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

        with open("output.txt", "w") as file:
            print(f"Start Of Simulation. Number Of steps : {num_steps}", file=file)
            for i in range(num_steps):
                print(f"Start Of Iteration number. {i} Parameters : ", file=file)
                print(f"Q : {self.Q}", file=file)
                print(f"R : {self.R}", file=file)
                print(f"m1x_posterior : {self.m1x_posterior}", file=file)
                print(f"m2x_posterior : {self.m2x_posterior}", file=file)
                print(f"target_est_state : {self.tgt_est_state}", file=file)
                print(f"target_Real_state : {self.tgt_real_state}", file=file)
                self.step(file=file)
                print(f"done step : {i}", file=file)

                target_real_traj = self.tgt_real_traj[i].squeeze().tolist()
                target_est_traj = self.tgt_est_traj[i].squeeze().tolist()
                target_delta_traj = torch.norm(self.tgt_real_traj[i].squeeze()-self.tgt_est_traj[i].squeeze())
                if target_delta_traj>10:
                    print("large delta")
                if torch.isnan(target_delta_traj):
                    print("nan")
                #print(f"delta estimation & actual {i + 1}: {target_delta_traj}", file=file)

                # Update the target position on the 3D plot
                ax.scatter(*target_real_traj, c="r", marker="o", label="Target Real" if i == 0 else "")
                # Update the target position on the 3D plot
                ax.scatter(*target_est_traj, c="g", marker="*", label="Target Estimated" if i == 0 else "")
                if i < len(self.tracker_traj):
                    tracker_traj = self.tracker_traj[i].squeeze().tolist()
                    #print(f"Tracker Step {i + 1}: {tracker_traj}", file=file)

                    # Update the tracker position on the 3D plot
                    ax.scatter(*tracker_traj, c="b", marker="^", label="Tracker" if i == 0 else "")

                # Update the plot every step and pause briefly
                if i == 0:
                    ax.legend()
                plt.pause(0.01)

            mse = calculate_loss(self.tgt_est_traj, self.tgt_real_traj)
            print("Mean Squared Error between est_state and real_state: ",mse)
            print(f"Mean Squared Error between est_state and real_state: {mse}", file=file)

        # Keep the plot open after the simulation
        #plt.show()



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
    model = MLP()
    model.initialize_weights()
    env.train_sequential(model, num_steps=1000)
    #env.train(model)
    #env.generate_simulation(num_steps=1000)
    print("Pipeline Ends")
    today = datetime.today()
    now = datetime.now()
    strToday = today.strftime("%m.%d.%y")
    strNow = now.strftime("%H:%M:%S")
    strTime = strToday + "_" + strNow
    print("Current Time =", strTime)



