import torch
from parameters import getJacobian
from utils import *
class ExtendedKalmanFilter:
    def __init__(self, SystemModel, device):
        self.f = SystemModel.f # a function that describes the state transition model, which maps the state at time t-1 to the state at time t.
        self.m = SystemModel.m # the dimensionality of the state variable.
        self.Q = SystemModel.Q # the covariance matrix of the process noise.
        self.h = SystemModel.h # a function that describes the measurement model, which maps the state at time t to the observation at time t.
        self.n = SystemModel.n # the dimensionality of the observation variable.
        self.R = SystemModel.R # the covariance matrix of the observation noise.
        self.m1x_posterior = SystemModel.m1x_0
        self.m2x_posterior=SystemModel.m2x_0
        # Device
        if device == 'cuda':
            self.device = torch.device('cuda')
        elif device == 'none':
            pass
        else:
            self.device = torch.device('cpu')
    # Predict
    def Predict(self , tracker_state):
        # Predict the 1-st moment of x
        self.m1x_prior = self.f(self.m1x_posterior)

        # Compute the Jacobians
        F = getJacobian(self.m1x_posterior,tracker_state , self.f)
        H = getJacobian(self.m1x_prior,tracker_state , self.h)
        self.jac_H = H
        # Predict the 2-nd moment of x
        self.m2x_prior = torch.matmul(F, torch.matmul(self.m2x_posterior, torch.transpose(F, 0, 1))) + self.Q

        # Predict the 1-st moment of y
        self.m1y = self.h(self.m1x_prior, tracker_state)

        # Predict the 2-nd moment of y
        self.m2y = torch.matmul(H, torch.matmul(self.m2x_prior, torch.transpose(H, 0, 1))) + self.R

    # Compute the Kalman Gain ULI Kt =Σt|t−1 ·H·S−1 t|t−1 .
    def KGain(self, tracker_state ):#, m2x_prior, tracker_state ):
        self.KG = torch.matmul(self.m2x_prior, torch.transpose(self.jac_H , 0, 1))
        self.KG = torch.matmul(self.KG, torch.inverse(self.m2y))

    # Innovation difference ∆yt = yt − ˆyt|t−1.
    def Innovation(self, y):
        self.dy = y - self.m1y

    # Compute Posterior ULI where in the article?
    def Correct(self):#, m1x_prior, m2x_prior):
        # Compute the 1-st posterior moment
        self.m1x_posterior =self.m1x_prior + torch.matmul(self.KG, self.dy)

        # Compute the 2-nd posterior moment
        self.m2x_posterior =self.m2x_prior - torch.matmul(self.KG, torch.matmul(self.m2y, torch.transpose(self.KG, 0, 1)))
        # Check if all eigenvalues are non-negative
        psd =is_psd((torch.inverse(self.m2x_posterior)[:3,:3]))
        # Print result
        if not psd:
            print("The matrix is not positive semi-definite.")

    def Update(self, y,tracker_state,file,x_temp):
        #FIXME: remove the bug that reduces tracker_state dimensionality
        tracker_state = torch.reshape(tracker_state, (6, 1))
        self.Predict(tracker_state)
        # print(f"done predict step, Parameters : ", file=file)
        # print(f"self.m1x_prior : {self.m1x_prior} ", file=file)
        # print(f"actual x : {x_temp} ", file=file)
        # print(f"self.m2x_prior : {self.m2x_prior} ", file=file)
        # print(f"self.m1y : {self.m1y} ", file=file)
        # print(f"self.m2y : {self.m2y} ", file=file)
        self.KGain(tracker_state)
        self.Innovation(y)
        self.Correct()
        # print(f"done correct step, Parameters : ", file=file)
        # print(f"self.m1x_posterior: {self.m1x_posterior} ", file=file)
        # print(f"actual x : {x_temp} ", file=file)
        # print(f"self.m2x_posterior : {self.m2x_posterior} ", file=file)
        # print(f"Done EKF ", file=file)
        return self.m1x_posterior, self.m2x_posterior,  self.m2x_prior,self.m2y, self.jac_H, self.KG
    #########################
    def UpdateJacobians(self, F, H):
        self.F = F.to(self.device)
        self.F_T = torch.transpose(F, 0, 1)
        self.H = H.to(self.device)
        self.H_T = torch.transpose(H, 0, 1)
#TODO: fix function
    def Init_batched_sequence(self, m1x_0, m2x_0):
        self.m1x_0 = m1x_0  # [ m, 1]
        self.m2x_0 = m2x_0  # [ m, m]


    def is_psd(self, mat):
        return bool((mat == mat.T).all() and (torch.eig(mat)[0][:,0]>=0).all())