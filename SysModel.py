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
        self.real_traj = []
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
    def GenerateStep(self, Q_gen, R_gen,tracker_state):
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
        if torch.equal(Q_gen, torch.zeros(self.m, self.m)):  # No noise
            xt = self.f(self.x_prev)
            self.real_traj.append(torch.reshape(xt,(self.m,1)))
        elif self.m == 1:  # 1 dim noise
            xt = self.f(self.x_prev)
            self.real_traj.append(torch.reshape(xt,(self.m,1)))
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
            self.real_traj.append(torch.reshape(xt, (self.m, 1)))
            ################
            ### Emission ###
            ################
            # yt = h(y)+n  #
            # if torch.isnan(xt).any():
            #     print("none")
            yt = self.h(xt, tracker_state)
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
            self.y = yt

            ################################
            ### Save Current to Previous ###
            ################################
            self.x_prev = xt
            return xt
            ##################################################################################################
            ######################## For DL module training process ##########################################
            ##################################################################################################

    #########################
    ### Generate Sequence ###
    #########################
    def GenerateSequence(self, Q_gen, R_gen, T):
        # Pre allocate an array for current state
        self.x = torch.zeros(size=[self.m, T])
        # Set x0 to be x previous
        self.x_prev = self.m1x_0
        xt = self.x_prev

        # Generate Sequence Iteratively
        for t in range(0, T):

            ########################
            #### State Evolution ###
            ########################
            if torch.equal(Q_gen, torch.zeros(self.m, self.m)):  # No noise
                xt = self.f(self.x_prev)
            else:
                xt = self.f(self.x_prev)
                mean = torch.zeros([self.m])
                distrib = MultivariateNormal(loc=mean, covariance_matrix=Q_gen)
                eq = distrib.rsample()
                eq = torch.reshape(eq[:], xt.size())
                # Additive Process Noise
                xt = torch.add(xt, eq)

            ########################
            ### Squeeze to Array ###
            ########################

            # Save Current State to Trajectory Array
            self.x[:, t] = torch.squeeze(xt, 1)

            ################################
            ### Save Current to Previous ###
            ################################
            self.x_prev = xt

    def Init_batched_sequence(self, m1x_0_batch, m2x_0_batch):

        self.m1x_0_batch = m1x_0_batch
        self.x_prev = m1x_0_batch
        self.m2x_0_batch = m2x_0_batch
    ######################
    ### Generate Batch ###
    ######################
    def GenerateBatch(self, size, T, randomInit=False, distribution="normal"):
        if (randomInit):
            # Allocate Empty Array for Random Initial Conditions
            self.m1x_0_rand = torch.zeros(size, self.m, 1)
            if distribution == 'uniform':
                ### if Uniform Distribution for random init
                for i in range(size):
                    initConditions = torch.rand_like(self.m1x_0) * args.variance
                    self.m1x_0_rand[i, :, 0:1] = initConditions.view(self.m, 1)

            elif distribution == 'normal':
                ### if Normal Distribution for random init
                for i in range(size):
                    distrib = MultivariateNormal(loc=torch.squeeze(self.m1x_0), covariance_matrix=self.m2x_0)
                    initConditions = distrib.rsample().view(self.m, 1)
                    self.m1x_0_rand[i, :, 0:1] = initConditions
            else:
                raise ValueError('args.distribution not supported!')

            self.Init_batched_sequence(self.m1x_0_rand, self.m2x_0)  ### for sequence generation
        else:  # fixed init
            initConditions = self.m1x_0.view(1, self.m, 1).expand(size, -1, -1)
            self.Init_batched_sequence(initConditions, self.m2x_0)  ### for sequence generation


        # Allocate Empty Array for Input
        self.Input = torch.empty(size, self.n, T)
        # Allocate Empty Array for Target
        self.Target = torch.empty(size, self.m, T)

        # Set x0 to be x previous
        self.x_prev = self.m1x_0_batch
        xt = self.x_prev

        # Generate in a batched manner
        for t in range(0, T):
            ########################
            #### State Evolution ###
            ########################
            if torch.equal(self.Q, torch.zeros(self.m, self.m)):  # No noise
                xt = self.f(self.x_prev)

            else:
                xt = self.f(self.x_prev)
                mean = torch.zeros([size, self.m])
                distrib = MultivariateNormal(loc=mean, covariance_matrix=self.Q)
                eq = distrib.rsample().view(size, self.m, 1)
                # Additive Process Noise
                xt = torch.add(xt, eq)

            ########################
            ### Squeeze to Array ###
            ########################

            # Save Current State to Trajectory Array
            self.Target[:, :, t] = torch.squeeze(xt, 2)

            ################################
            ### Save Current to Previous ###
            ################################
            self.x_prev =xt
