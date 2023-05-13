import math as mt
import numpy as np
import SysModel
import torch

class UAV:
    def __init__(self,state):
        self.state = state
        self.current_position = self.state[:3, 0]
        self.current_velocity = self.state[3:, 0]
        self.delta_t = 1  # The size of discrete time step in seconds

    def Update_state(self,state):
        self.state = state
        self.current_position = self.state[:3, 0]
        self.current_velocity = self.state[3:, 0]

    def next_position(self, v, heading, tilt):
        """
        A method calculating the next position of a tracker given its control step {v,heading,tilt} and current position {x,y,z}
        :param target_state: [[x,vx],[y,vy],[z,vz]]
        :return:self.current_position: [[x], [y], [z]]
                self.current_velocity = [[vx], [vy], [vz]]
        """
        # Calculate the control step based on target state estimation
        v, heading, tilt = v, heading, tilt

        # Next position calculation based on current position, velocity, and control inputs
        x = self.current_position[0] + v * (torch.cos(heading)) * (torch.sin(tilt)) * self.delta_t
        y = self.current_position[1] + v * (torch.sin(heading)) * (torch.sin(tilt)) * self.delta_t
        z = self.current_position[2] + v * (torch.sin(tilt)) * self.delta_t

        vx = (x - self.current_position[0]) / self.delta_t
        vy = (y - self.current_position[1]) / self.delta_t
        vz = (z - self.current_position[2]) / self.delta_t

        self.current_position = torch.tensor([[x], [y], [z]])
        self.current_velocity = torch.tensor([[vx], [vy], [vz]])

        # Document the Tracker trajectory
        self.state = torch.cat((self.current_position, self.current_velocity), dim=0)
        return self.state