import math as mt
import numpy as np

class UAV:
    def __init__(self, position =np.array([[0],[0],[0]]) , velocity = np.array([[0],[0],[0]])):
        self.current_position = position
        self.current_velocity = velocity
        self.position_list = [position]
        self.velocity_list = [velocity]
        # TODO: Implement delta_t
        self.delta_t = 1    # The size of descrete time step in seconds

class Tracker(UAV):
    def __init__(self,position = np.array([[0],[0],[0]]), velocity = np.array([[0],[0],[0]])):
        super().__init__(position, velocity)
        self.velocity_magnitude = 0
        self.heading = 0
        self.tilt = 0

    def control_step(self, target_position):
        """
        A method calculating the control step of a tracker given the target state estimation
        :param target_state: [[x,vx],[y,vy],[z,vz]]
        :return: v: velocity magnitude
                heading: angle
                tilt: angle
        """
        # TODO: Implement control logic to calculate v, heading, and tilt
        # Return constant values for now
        # required flow: input: observation---> kalman filter---> loss---> return control decision
        self.velocity_magnitude = 0.6
        self.heading = 50
        self.tilt = 50
        return self.velocity_magnitude, self.heading, self.tilt


    def next_position(self):
        """
        A method calculating the next position of a tracker given its control step {v,heading,tilt} and current position {x,y,z}
        :param target_state: [[x,vx],[y,vy],[z,vz]]
        :return:self.current_position: [[x], [y], [z]]
                self.current_velocity = [[vx], [vy], [vz]]
        """
        # Calculate the control step based on target state estimation
        v, heading, tilt = self.control_step(target_position=self.current_position)

        # Next position calculation based on current position, velocity, and control inputs
        x = self.current_position[0][0] + v*(mt.cos(heading))*(mt.sin(tilt))*self.delta_t
        y = self.current_position[1][0] + v*(mt.sin(heading))*(mt.sin(tilt))*self.delta_t
        z = self.current_position[2][0] + v*(mt.sin(tilt))*self.delta_t

        vx = (x-self.current_position[0])/self.delta_t
        vy = (y-self.current_position[1])/self.delta_t
        vz = (z-self.current_position[2])/self.delta_t

        self.current_position = [[x], [y], [z]]
        self.current_velocity = [[vx], [vy], [vz]]

        # Document the Tracker trajectory
        self.position_list.append(self.current_position)
        self.velocity_list.append(self.current_velocity)



class Target(UAV):
    def __init__(self, position =np.array([[0], [0], [90]]), velocity = np.array([[-0.3], [0.4], [0]])):
        super().__init__(position, velocity)
        self.current_position = position
        self. current_velocity = velocity

        # Composing the state from position and velocity
        self.current_state = np.concatenate((self.current_position, self.current_velocity), axis=0)

        # Calculating the state matrix
        A = np.identity(3)
        B = np.identity(3) * self.delta_t
        C = np.zeros((3, 3))
        D = np.identity(3)
        top = np.concatenate((A, B), axis=1)
        bottom = np.concatenate((C, D), axis=1)

        self.state_matrix = np.concatenate((top, bottom), axis=0)

        # Calculating the noise covariance matrix , constants are from the use case in the original paper
        diagonal = [(1e-5), (1e-5), (1e-6)] #2
        diagonal_matrix = np.diag(diagonal)
        A = ((self.delta_t^3)/3)*diagonal_matrix
        B = ((self.delta_t^2)/2)*diagonal_matrix
        C = ((self.delta_t^2)/2)*diagonal_matrix
        D = (self.delta_t)*diagonal_matrix
        top = np.concatenate((A, B), axis=1)
        bottom = np.concatenate((C, D), axis=1)
        #TODO : EIGENVALUES ARE NEGATIVE
        self.Q = np.concatenate((top, bottom), axis=0)

    def state_step(self):
        # Generate random sample from a normal distribution with mean=0 and covariance=Q
        samples = np.transpose(np.random.multivariate_normal(mean=np.zeros(6), cov=self.Q, size=1))

        # Dynamical model
        state = (np.dot(self.state_matrix, self.current_state))+samples

        self.current_state = state
        self.current_position = self.current_state[:3]
        self.current_velocity = self.current_state[-3:]

        self.current_state = state

    def next_position(self):
        pass

    def next_velocity(self):
        pass