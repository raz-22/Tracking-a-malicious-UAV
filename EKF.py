import torch
from parameters import getJacobian

class ExtendedKalmanFilter:
    def __init__(self, f, m, Q, h, n, R):
        self.f = f # a function that describes the state transition model, which maps the state at time t-1 to the state at time t.
        self.m = m # the dimensionality of the state variable.
        self.Q = Q # the covariance matrix of the process noise.
        self.h = h # a function that describes the measurement model, which maps the state at time t to the observation at time t.
        self.n = n # the dimensionality of the observation variable.
        self.R = R # the covariance matrix of the observation noise.

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

