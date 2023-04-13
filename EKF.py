"""# **Class: Extended Kalman Filter**
Theoretical Non Linear Kalman
This code defines a class for an Extended Kalman Filter (EKF) in PyTorch,
 which is a popular algorithm for state estimation in non-linear dynamic systems.
The EKF is a recursive algorithm that estimates the state of a system using a sequence of measurements.
 At each time step, the EKF predicts the current state of the system based on its previous estimate and the known system dynamics, 
 and then adjusts its estimate based on the current measurement.

 The class ExtendedKalmanFilter contains several methods that implement the various steps of the EKF algorithm. The key methods are:

Predict(): This method predicts the state of the system at the current time step, based on the previous estimate of the state and the system dynamics.
It also computes the Jacobians of the system dynamics and measurement functions.
KGain(): This method computes the Kalman gain, which is used to adjust the state estimate based on the current measurement.
Innovation(): This method computes the difference between the predicted measurement and the actual measurement, which is used to update the state estimate.
Correct(): This method updates the state estimate based on the innovation and the Kalman gain.
The Update(): method combines all of the above steps into a single update step of the EKF algorithm.

The GenerateBatch() method generates estimates of the state of the system and its covariance for a batch of input sequences. 
The input sequence is a batch of observations of the system, and the output is a batch of estimates of the state and covariance of the system at each time step.
"""
import torch

from parameters import getJacobian


class ExtendedKalmanFilter:
    def __init__(self, f, m, Q, h, n, R, T):
        self.f = f # a function that describes the state transition model, which maps the state at time t-1 to the state at time t.
        self.m = m # the dimensionality of the state variable.
        self.Q = Q # the covariance matrix of the process noise.
        self.h = h # a function that describes the measurement model, which maps the state at time t to the observation at time t.
        self.n = n # the dimensionality of the observation variable.
        self.R = R # the covariance matrix of the observation noise.
        self.T = T # the length of the sequence of observations to be processed.
    # Predict
    def Predict(self, m1x_posterior, m2x_posterior):
        # Predict the 1-st moment of x
        self.m1x_prior = self.f(m1x_posterior)
        # Compute the Jacobians
        self.UpdateJacobians(getJacobian(m1x_posterior, self.f), getJacobian(self.m1x_prior, self.h))
        # Predict the 2-nd moment of x
        self.m2x_prior = torch.bmm(self.batched_F, m2x_posterior)#This line performs a batch matrix multiplication between the Jacobian matrix (batched_F) and the 2nd order moments of the prior state estimate (m2x_posterior), resulting in a predicted 2nd order moment of the prior state estimate (m2x_prior). The batch matrix multiplication is performed using PyTorch's torch.bmm function.
        self.m2x_prior = torch.bmm(self.m2x_prior, self.batched_F_T) + self.Q

        # Predict the 1-st moment of y
        self.m1y = self.h(self.m1x_prior)
        # Predict the 2-nd moment of y
        self.m2y = torch.bmm(self.batched_H, self.m2x_prior)
        self.m2y = torch.bmm(self.m2y, self.batched_H_T) + self.R

    # Compute the Kalman Gain ULI Kt =Σt|t−1 ·H·S−1 t|t−1 .
    def KGain(self, m2x_prior):
        self.KG = torch.bmm(m2x_prior, self.batched_H_T)
        self.KG = torch.bmm(self.KG, torch.inverse(self.m2y))

        # Save KalmanGain
        self.KG_array[:, :, :, self.i] = self.KG
        self.i += 1

    # Innovation difference ∆yt = yt − ˆyt|t−1.
    def Innovation(self, y):
        self.dy = y - self.m1y

    # Compute Posterior ULI where in the article?
    def Correct(self, m1x_prior, m2x_prior):
        # Compute the 1-st posterior moment
        self.m1x_posterior = m1x_prior + torch.bmm(self.KG, self.dy)

        # Compute the 2-nd posterior moment
        self.m2x_posterior = torch.bmm(self.m2y, torch.transpose(self.KG, 1, 2))
        self.m2x_posterior = m2x_prior - torch.bmm(self.KG, self.m2x_posterior)

    def Update(self, y, m1x_posterior, m2x_posterior):
        self.Predict(m1x_posterior, m2x_posterior)
        self.KGain(m2x_posterior)
        self.Innovation(y)
        self.Correct(self.m1x_prior, self.m2x_prior)

        return self.m1x_posterior, self.m2x_posterior

    #########################
    #The EKF linearizes the differentiable f (·) and h(·) in a time-dependent manner using their partial derivative xt|t−1 . Namely, matrices, also known as Jacobians, evaluated at ˆxt−1|t−1 and ˆ ˆ Ft =Jf ˆxt−1|t−1 ˆ Ht =Jhˆxt|t−1,
    def UpdateJacobians(self, F, H):
        self.batched_F = F
        self.batched_F_T = torch.transpose(F, 1, 2)
        self.batched_H = H
        self.batched_H_T = torch.transpose(H, 1, 2)

    def Init_batched_sequence(self, m1x_0_batch, m2x_0_batch):
        self.m1x_0_batch = m1x_0_batch  # [batch_size, m, 1]
        self.m2x_0_batch = m2x_0_batch  # [batch_size, m, m]

    ######################
    ### Generate Batch ###
    ######################
    def GenerateBatch(self, y):
        """
        input y: batch of observations [batch_size, n, T]
        """
        self.batch_size = y.shape[0] # batch size
        T = y.shape[2] # sequence length (maximum length if randomLength=True)

        # Pre allocate KG array
        self.KG_array = torch.zeros([self.batch_size,self.m,self.n,T])
        self.i = 0 # Index for KG_array allocation

        # Allocate Array for 1st and 2nd order moments (use zero padding)
        self.x = torch.zeros(self.batch_size, self.m, T)
        self.sigma = torch.zeros(self.batch_size, self.m, self.m, T)

        # Set 1st and 2nd order moments for t=0
        self.m1x_posterior = self.m1x_0_batch
        self.m2x_posterior = self.m2x_0_batch

        # Generate in a batched manner
        for t in range(0, T):
            yt = torch.unsqueeze(y[:, :, t], 2)
            xt, sigmat = self.Update(yt,self.m1x_posterior,self.m2x_posterior)
            self.x[:, :, t] = torch.squeeze(xt, 2)
            self.sigma[:, :, :, t] = sigmat
            
        return self.KG_array, self.x, self.sigma, self.m1x_posterior, self.m2x_posterior