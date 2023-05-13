"""# **Class: Extended Kalman Filter**
Theoretical Non Linear Kalman
"""
import torch

from Parameters import getJacobian
from Utils import *


class ExtendedKalmanFilter:

    def __init__(self, SystemModel, usecuda=False):
        # Device
        if usecuda:
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        # process model
        self.f = SystemModel.f
        self.m = SystemModel.m
        self.Q = SystemModel.Q.to(self.device)
        self.m1x_posterior = SystemModel.m1x_0
        self.m2x_posterior=SystemModel.m2x_0
        # observation model
        self.h = SystemModel.h
        self.n = SystemModel.n
        self.R = SystemModel.R.to(self.device)
        self.step =0
        # sequence length (use maximum length if random length case)
        # self.T = SystemModel.T
        # self.T_test = SystemModel.T_test

    # Predict
    def Predict(self , tracker_state):
        # Predict the 1-st moment of x
        self.m1x_prior = self.f(self.m1x_posterior).to(self.device)
        # Compute the Jacobians
        self.UpdateJacobians(getJacobian(self.m1x_posterior,tracker_state , self.f), getJacobian(self.m1x_prior,tracker_state , self.h))
        # Predict the 2-nd moment of x
        self.m2x_prior = torch.matmul(self.batched_F, self.m2x_posterior)
        self.m2x_prior = torch.matmul(self.m2x_prior, self.batched_F_T) + self.Q

        ####### Only For TEST Purposes ####
        ###################################
        # Check if all eigenvalues are non-negative
        symetric = is_symetric(self.m2x_prior)
        psd, symetric_2 = is_psd((torch.inverse(self.m2x_prior))[:3, :3])

        # Print result
        if not symetric:
            raise ValueError("EKF: step: " + str(self.step) + " the prior matrix is not symetric")
        if not psd:
            raise ValueError("EKF: step: " + str(self.step) + " The prior matrix is not positive semi-definite.")
        ###################################

        # Predict the 1-st moment of y
        self.m1y = self.h(self.m1x_prior, tracker_state)

        # Predict the 2-nd moment of y
        self.m2y = torch.matmul(self.batched_H, self.m2x_prior)
        self.m2y = torch.matmul(self.m2y, self.batched_H_T) + self.R

    # Compute the Kalman Gain
    def KGain(self):
        self.KG = torch.matmul(self.m2x_prior, self.batched_H_T)
        self.KG = torch.matmul(self.KG, torch.inverse(self.m2y))

    # Innovation
    def Innovation(self, y):
        self.dy = y - self.m1y

    # Compute Posterior
    def Correct(self):
        # Compute the 1-st posterior moment
        self.m1x_posterior = self.m1x_prior + torch.matmul(self.KG, self.dy)

        # Compute the 2-nd posterior moment
        if self.KG.dim() == 2:
            self.m2x_posterior = torch.matmul(self.m2y, torch.transpose(self.KG, 0, 1))
            self.m2x_posterior = self.m2x_prior - torch.matmul(self.KG, self.m2x_posterior)
            ###################################
            ####### Only For TEST Purposes ####
            ###################################
            # Check if all eigenvalues are non-negative
            postsymetric=is_symetric(self.m2x_posterior)
            symetric = bool(torch.equal(self.m2x_posterior, self.m2x_posterior.T))
            print(self.m2x_posterior-self.m2x_posterior.T)
            print(torch.linalg.eig(self.m2x_posterior).eigenvalues)

            postpsd,postsymetric_2 = is_psd((torch.inverse(self.m2x_posterior))[:3, :3])
            # Print result
            if not postsymetric:
                raise ValueError("EKF: step: " +str(self.step)+" The posterior matrix is not symetric")
            if not postpsd:
                raise ValueError("EKF: step: " +str(self.step)+" The posterior matrix is not positive semi-definite.")

        else:
            self.m2x_posterior = torch.bmm(self.m2y, torch.transpose(self.KG, 0, 1))
            self.m2x_posterior = self.m2x_prior - torch.bmm(self.KG, self.m2x_posterior)

    def Update(self,tracker_state, y,step=0):
        self.step = step
        # FIXME: remove the bug that reduces tracker_state dimensionality
        self.Predict(tracker_state)
        self.KGain()
        self.Innovation(y)
        self.Correct()


    #########################

    def UpdateJacobians(self, F, H):
        if F.dim()==4:
            self.batched_F = F.to(self.device)
            self.batched_F_T = torch.transpose(F, 1, 2)
            self.batched_H = H.to(self.device)
            self.batched_H_T = torch.transpose(H, 1, 2)
        else:
            self.batched_F = F.to(self.device)
            self.batched_F_T = torch.transpose(F, 0, 1)
            self.batched_H = H.to(self.device)
            self.batched_H_T = torch.transpose(H, 0, 1)

    def Init_batched_sequence(self, m1x_0_batch, m2x_0_batch):

        self.m1x_0_batch = m1x_0_batch  # [batch_size, m, 1]
        self.m2x_0_batch = m2x_0_batch  # [batch_size, m, m]