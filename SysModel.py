"""# **Class: System Model for Non-linear Cases**

1 Store system model parameters:
    state transition function f,
    observation function h,
    process noise Q,
    observation noise R,
    train&CV dataset sequence length T,
    test dataset sequence length T_test,
    state dimension m,
    observation dimension n, etc.

2 Generate datasets for non-linear cases
"""

import torch
from torch.distributions.multivariate_normal import MultivariateNormal


class SystemModel:
    def __init__(self, f, Q, h, R, m, n, T, T_test, prior_Q=None, prior_Sigma=None, prior_S=None):

        ####################
        ### Motion Model ###
        ####################
        self.f = f
        self.m = m
        self.Q = Q
        #########################
        ### Observation Model ###
        #########################
        self.h = h
        self.n = n
        self.R = R
        ################
        ### Sequence ###
        ################
        # Assign T
        self.T = T
        self.T_test = T_test

        #########################
        ### Covariance Priors ###
        #########################
        if prior_Q is None:
            self.prior_Q = torch.eye(self.m)
        else:
            self.prior_Q = prior_Q

        if prior_Sigma is None:
            self.prior_Sigma = torch.zeros((self.m, self.m))
        else:
            self.prior_Sigma = prior_Sigma

        if prior_S is None:
            self.prior_S = torch.eye(self.n)
        else:
            self.prior_S = prior_S

    #####################
    ### Init Sequence ###
    #####################
    def InitSequence(self, m1x_0, m2x_0):

        self.m1x_0 = m1x_0
        self.m2x_0 = m2x_0

    #########################
    ### Update Covariance ###
    #########################
    def UpdateCovariance_Matrix(self, Q, R):

        self.Q = Q

        self.R = R

    #########################
    ### Generate Sequence ###
    #########################
    def GenerateStep(self, Q_gen, R_gen,tracker_state,target_state):
        """
        for T=1, our case of UAV tracking problem, the method has:
        input: Q,R covariance matrices
        output: updating 1 state step to the observation and state arrays of using the system dynamical model
        """
        # Pre allocate an array for current state
        self.x = torch.zeros(size=[self.m, 1])
        # Pre allocate an array for current observation
        self.y = torch.zeros(size=[self.n, 1])

        # Set x0 to be x previous
        self.x_prev = self.m1x_0
        xt = self.x_prev
        ########################
        #### State Evolution ###
        ########################
        ##### xt = f(xt)+q  #####
        if torch.equal(Q_gen, torch.zeros(self.m, self.m)):  # No noise
            xt = self.f(self.x_prev)
        elif self.m == 1:  # 1 dim noise
            xt = self.f(self.x_prev)
            eq = torch.normal(mean=0, std=Q_gen)
            # Additive Process Noise
            xt = torch.add(xt, eq)
        else:
            xt = self.f(self.x_prev)
            mean = torch.zeros([self.m])
            distrib = MultivariateNormal(loc=mean, covariance_matrix=Q_gen)
            eq = distrib.rsample()
            eq = torch.reshape(eq[:], xt.size())
            # Additive Process Noise
            xt = torch.add(xt, eq)

            ################
            ### Emission ###
            ################
            # yt = h(y)+n  #

            yt = self.h(xt, target_state, tracker_state)
            # Observation Noise
            if self.n == 1:  # 1 dim noise
                er = torch.normal(mean=0, std=R_gen)
                # Additive Observation Noise
                yt = torch.add(yt, er)
            else:
                mean = torch.zeros([self.n])
                distrib = MultivariateNormal(loc=mean, covariance_matrix=R_gen)
                er = distrib.rsample()
                er = torch.reshape(er[:], yt.size())
                # Additive Observation Noise
                yt = torch.add(yt, er)

            ########################
            ### Squeeze to Array ###
            ########################

            # Save Current State to Trajectory Array
            self.x=xt

            # Save Current Observation to Trajectory Array
            self.y = yt.unsqueeze(1)

            ################################
            ### Save Current to Previous ###
            ################################
            self.x_prev = xt
