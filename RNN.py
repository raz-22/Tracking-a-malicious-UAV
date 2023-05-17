import torch
import torch.nn as nn

class ElmanRNN(nn.Module):
    def __init__(self, input_size=12, hidden_size=64, output_size=3):
        super(ElmanRNN, self).__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=False)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, target_state, tracker_state):
        x = torch.cat((target_state.squeeze(), tracker_state.squeeze()), dim=0).reshape(1,1,input_size)
        hidden = self.init_hidden(1)  # Set batch size to 1
        output, _ = self.rnn(x, hidden)
        output = output[:, -1, :]  # Taking the last time step output
        output = self.fc(output)
        return output.reshape(3,1)

    def init_hidden(self,batch_size):
        return torch.zeros(1,batch_size,  self.hidden_size)

# # Define the network dimensions
# input_size = 12
# hidden_size = 8
# output_size = 3
# ### initial Tracker State ###
# tracker_state = torch.reshape(torch.tensor([[50.], [50.], [80.], [-3.], [4.], [4.]]), (6,1))
#
# ### Initial State 1ST and 2ND Moments ###
# m1x_0 = torch.reshape(torch.tensor([[10.], [10.], [90.], [-3.], [4.], [4.]]), (6,1))
# # Create an instance of the ElmanRNN
# elman_rnn = ElmanRNN(input_size, hidden_size, output_size)
#
# # Create a random input tensor
# input_tensor = torch.randn(12,1)
#
# # Forward pass
# output = elman_rnn.forward(m1x_0,tracker_state)
#
# print(output.shape)  # Should print torch.Size([1, 3, 1])