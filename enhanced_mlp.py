import torch
import torch.nn as nn
import torch.nn.init as init


class MLP_memory(nn.Module):
    def __init__(self, hidden_size=64):
        super(MLP_memory, self).__init__()

        self.hidden_size = hidden_size

        # Define the two hidden layers
        self.fc1 = nn.Linear(18, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)

        # Define the output layer
        self.fc3 = nn.Linear(hidden_size, 3)
    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
    def forward(self, target_state,memory_state, tracker_state):
        x = torch.cat((target_state.squeeze(),memory_state.squeeze()), dim=0)
        x= torch.cat((x.squeeze(),tracker_state.squeeze()),dim = 0)
        # Pass the input through the hidden layers with ReLU activation
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))

        # Pass the output through the output layer with linear activation
        x = self.fc3(x)
        return x