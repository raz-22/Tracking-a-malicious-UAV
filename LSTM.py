import torch
import torch.nn as nn

class ElmanLSTM(nn.Module):
    def __init__(self, input_size=12, hidden_size=64, output_size=3):
        super(ElmanLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.input_size = input_size
    def forward(self, target_state, tracker_state):
        x = torch.cat((target_state.squeeze(), tracker_state.squeeze()), dim=0).reshape(1,1,self.input_size)
        hidden = self.init_hidden(1)  # Set batch size to 1
        _, (output, _) = self.lstm(x, hidden)
        output = output.squeeze(0)
        output = self.fc(output)
        return output.reshape(3,1)

    def init_hidden(self, batch_size):
        return (torch.zeros(1, batch_size, self.hidden_size),
                torch.zeros(1, batch_size, self.hidden_size))
#
# # Define the network dimensions
# input_size = 12
# hidden_size = 8
# output_size = 3
#
# # Initial Tracker State
# tracker_state = torch.reshape(torch.tensor([[50.], [50.], [80.], [-3.], [4.], [4.]]), (6, 1))
#
# # Initial State 1st and 2nd Moments
# m1x_0 = torch.reshape(torch.tensor([[10.], [10.], [90.], [-3.], [4.], [4.]]), (6, 1))
#
# # Create an instance of the ElmanLSTM
# elman_lstm = ElmanLSTM(input_size, hidden_size, output_size)
#
# # Create a random input tensor
# input_tensor = torch.randn(12, 1)
#
# # Forward pass
# output = elman_lstm.forward(m1x_0, tracker_state)
#
# print(output.shape)  # Should print torch.Size([1, 3])