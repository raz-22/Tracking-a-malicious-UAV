from UAV import *
import math as mt

import numpy as np
from numpy import linspace
from utils import *
import matplotlib; matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits import mplot3d

class Environment:
    def __init__(self):
        self.tracker = Tracker()
        self.target = Target()
        self.delta_t = 1
        # Calculating the state matrix
        A = np.identity(3)
        B = np.identity(3) * self.delta_t
        C = np.zeros((3, 3))
        D = np.identity(3)
        top = np.concatenate((A, B), axis=1)
        bottom = np.concatenate((C, D), axis=1)

        self.state_matrix = np.concatenate((top, bottom), axis=0)

    def step(self):
        self.target.state_step()
        self.Observation()
        #self.EKF()
        #self.control_step()
        self.tracker.next_position()

    def Observation(self, gamma=4, lamda=(3.8961*(1e-3))):
        """
        A method calculating
        :param
        :return:
        """
        los = self.target.current_position-self.tracker.current_position

        delta_x = los[0]
        delta_y = los[1]
        delta_z = los[2]

        #h(s) observstion function measurement equations
        gamma_d_2 = (gamma/2)*(np.linalg.norm(los))
        azimuth = np.arctan(delta_y/delta_x)
        elevation = np.arctan(delta_z / np.linalg.norm(los))
        #TODO: validate the radian velocity formula
        radian_velocity = np.dot(np.transpose(self.tracker.current_velocity), los)/(np.linalg.norm(los))

        doppler_shift = (4*radian_velocity)/(2*lamda)
        h = [gamma_d_2, azimuth, elevation, doppler_shift]

        #TODO: z_k=h(s)+n_k calculate the noise
        n_k = 0
        z = h # +n_k

    def EKF(self):
        #Prediction
        pred_s = np.dot(self.state_matrix,self.target.current_state)
        pred_m = np.dot(self.state_matrix,)



    def control_step(self, target_position):
        """
        A method calculating the control step of a tracker given the target state estimation
        :param target_state: [[x,vx],[y,vy],[z,vz]]
        :return: v: velocity magnitude
                heading: angle
                tilt: angle
        """
        # TODO: Implement control logic to calculate v, heading, and tilt
        # EKF-estimate state
        #Calculate loss
        #decide control step
        # Return constant values for now
        # required flow: input: observation---> kalman filter---> loss---> return control decision
        self.tracker.velocity_magnitude = 0.6
        self.tracker.heading = 50
        self.tracker.tilt = 50
        return self.velocity_magnitude, self.heading, self.tilt

    def generate_simulation(self, num_steps=1000):
        # Create a list to store the last 10 positions
        coords = []

        for i in range(num_steps):
            env.step()
            coords.append(env.target.current_position[:, 0])
        coords = np.array(coords)
        #coords = np.random.rand(100, 3)
        # Create a figure and 3D axes
        fig = plt.figure()
        ax = plt.axes(111, projection='3d')
        # Create tail

        tail, = ax.plot([], [], [], c='r')
        # Create newest dot
        dot, = ax.plot([], [], [], 'o', c='b')

        # Set axis limits based on min/max values in coords
        ax.set_xlim3d(coords[:, 0].min(), coords[:, 0].max())
        ax.set_ylim3d(coords[:, 1].min(), coords[:, 1].max())
        ax.set_zlim3d(coords[:, 2].min(), coords[:, 2].max())

        # Set animation function
        ani = animation.FuncAnimation(fig, update_graph, len(coords), fargs=(coords, tail,dot),interval=50) #, tail), interval=50)

        # Show plot
        plt.show()

if __name__ == '__main__':
    env = Environment()
    env.generate_simulation()

