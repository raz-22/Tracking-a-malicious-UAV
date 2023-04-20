import math as mt
import numpy as np
import SysModel
import torch

class UAV:
    def __init__(self,state):
        self.current_position = state[:3, 0]
        self.current_velocity = state[3:, 0]
        self.state = torch.cat((self.current_position, self.current_velocity) , dim = 0)
        self.position_list = [state[:3, 0]]
        self.velocity_list = [state[3:, 0]]
        self.delta_t = 1  # The size of discrete time step in seconds

    def Update_state(self,state):
        self.state = state
        self.current_position = state[:3, 0]
        self.current_velocity = state[3:, 0]
class Tracker(UAV):
    def __init__(self, state):
        super().__init__(state)
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
        self.velocity_magnitude = torch.tensor(0.6)
        self.heading = torch.tensor(50)
        self.tilt = torch.tensor(50)
        return self.velocity_magnitude, self.heading, self.tilt

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
        self.position_list.append(self.current_position)
        self.velocity_list.append(self.current_velocity)
        self.state = torch.cat((self.current_position, self.current_velocity), dim=0)
        return self.state

class Target(UAV):
    def __init__(self, state):
        super().__init__(state)


