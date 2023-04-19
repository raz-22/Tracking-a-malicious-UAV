import torch
from parameters import getJacobian

class ExtendedKalmanFilter:
    def __init__(self, SystemModel, device):
        self.f = SystemModel.f # a function that describes the state transition model, which maps the state at time t-1 to the state at time t.
        self.m = SystemModel.m # the dimensionality of the state variable.
        self.Q = SystemModel.Q # the covariance matrix of the process noise.
        self.h = SystemModel.h # a function that describes the measurement model, which maps the state at time t to the observation at time t.
        self.n = SystemModel.n # the dimensionality of the observation variable.
        self.R = SystemModel.R # the covariance matrix of the observation noise.
        # Device
        if device == 'cuda':
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
    # Predict
    def Predict(self, m1x_posterior, m2x_posterior):
        # Predict the 1-st moment of x
        self.m1x_prior = self.f(m1x_posterior)

        # Compute the Jacobians
        F = getJacobian(m1x_posterior, self.f)
        H = getJacobian(self.m1x_prior, self.h)

        # Predict the 2-nd moment of x
        self.m2x_prior = torch.bmm(F, torch.bmm(m2x_posterior, torch.transpose(F, 1, 2))) + self.Q

        # Predict the 1-st moment of y
        self.m1y = self.h(self.m1x_prior)

        # Predict the 2-nd moment of y
        self.m2y = torch.bmm(H, torch.bmm(self.m2x_prior, torch.transpose(H, 1, 2))) + self.R

    # Compute the Kalman Gain ULI Kt =Σt|t−1 ·H·S−1 t|t−1 .
    def KGain(self, m2x_prior):
        self.KG = torch.bmm(m2x_prior, torch.transpose(getJacobian(self.m1x_prior, self.h), 1, 2))
        self.KG = torch.bmm(self.KG, torch.inverse(self.m2y))

    # Innovation difference ∆yt = yt − ˆyt|t−1.
    def Innovation(self, y):
        self.dy = y - self.m1y

    # Compute Posterior ULI where in the article?
    def Correct(self, m1x_prior, m2x_prior):
        # Compute the 1-st posterior moment
        self.m1x_posterior = m1x_prior + torch.bmm(self.KG, self.dy)

        # Compute the 2-nd posterior moment
        self.m2x_posterior = m2x_prior - torch.bmm(self.KG, torch.bmm(self.m2y, torch.transpose(self.KG, 1, 2)))

    def Update(self, y, m1x_posterior, m2x_posterior):
        self.Predict(m1x_posterior, m2x_posterior)
        self.KGain(m2x_posterior)
        self.Innovation(y)
        self.Correct(self.m1x_prior, self.m2x_prior)

        return self.m1x_posterior, self.m2x_posterior

    #########################

    def UpdateJacobians(self, F, H):
        self.F = F.to(self.device)
        self.F_T = torch.transpose(F, 1, 2)
        self.H = H.to(self.device)
        self.H_T = torch.transpose(H, 1, 2)

#TODO: fix function
    def Init_batched_sequence(self, m1x_0, m2x_0):
        self.m1x_0 = m1x_0  # [ m, 1]
        self.m2x_0 = m2x_0  # [ m, m]

        ######################
        ### Generate Batch ###
        ######################

    ######################
    ### Compute Priors ###
    ######################
    def step_prior(self):
        # Predict the 1-st moment of x
        self.m1x_prior = self.f(self.m1x_posterior)

        # Predict the 1-st moment of y
        self.m1y = self.h(self.m1x_prior)

    ##############################
    ### Kalman Gain Estimation ###
    ##############################

    def step(self,y):
        # Compute Priors
        self.step_prior()

        # Compute Kalman Gain
        self.step_KGain_est(y)

        # Save KGain in array
        self.KGain_array[self.i] = self.KGain
        self.i += 1

        # Innovation
        y_obs = torch.unsqueeze(y, 1)
        dy = y_obs - self.m1y

        # Compute the 1-st posterior moment
        INOV = torch.matmul(self.KGain, dy)
        self.m1x_posterior_previous = self.m1x_posterior
        self.m1x_posterior = self.m1x_prior + INOV

        #self.state_process_posterior_0 = self.state_process_prior_0
        self.m1x_prior_previous = self.m1x_prior

        # update y_prev
        self.y_previous = y

        # return
        return torch.squeeze(self.m1x_posterior)