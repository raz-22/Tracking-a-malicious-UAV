
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
        self.x_prev = None
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

    def UpdateCovariance_Matrix(self, Q, R):
        self.Q = Q

        self.R = R


    #####################
    ### Init Sequence ###
    #####################
    def InitSequence(self, m1x_0, m2x_0):

        self.m1x_0 = m1x_0
        self.m2x_0 = m2x_0


    #########################
    ### Generate Sequence ###
    #########################
    def GenerateStep(self,tracker_state):
        """
        for T=1, our case of UAV tracking problem, the method has:
        input: Q,R covariance matrices
        output: updating 1 state step to the observation and state arrays of using the system dynamical model
        """
        #FIXME: remove the bug that reduces tracker_state dimensionality
        tracker_state = torch.reshape(tracker_state, (self.m, 1))
        # Pre allocate an array for current state
        self.x = torch.zeros(size=[self.m, 1])
        # Pre allocate an array for current observation
        self.y = torch.zeros(size=[self.n, 1])
        if self.x_prev is None:
            # Set x0 to be x previous
            self.x_prev = self.m1x_0
        xt = self.x_prev
        ########################
        #### State Evolution ###
        ########################
        ##### xt = f(xt)+q  #####
        xt = self.f(self.x_prev)
        mean = torch.zeros([self.m])
        distrib = MultivariateNormal(loc=mean, covariance_matrix=self.Q )
        eq = distrib.rsample()
        eq = torch.reshape(eq[:], xt.size())
        # Additive Process Noise
        xt = torch.add(xt, eq)

        ################
        ### Emission ###
        ################
        # yt = h(y)+n  #
        yt = self.h(xt, tracker_state)
        mean = torch.zeros([self.n])

        los = xt[:3, 0] - tracker_state[:3, 0]
        d_4 =(torch.norm(los))
        diagonal = [d_4, d_4, d_4, d_4]
        diagonal_matrix = torch.diag(torch.tensor(diagonal))
        R = torch.eye(4) * diagonal_matrix
        self.UpdateCovariance_Matrix(self.Q, R)



        distrib = MultivariateNormal(loc=mean, covariance_matrix=self.R )
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
        self.y = yt

        ################################
        ### Save Current to Previous ###
        ################################
        self.x_prev = xt
        return self.Q, R