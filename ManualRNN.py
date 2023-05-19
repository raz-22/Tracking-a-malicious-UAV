import torch
import torch.nn as nn

class RNN(nn.Module):
    def __init__(self, input_size=12, hidden_size=64, output_size=3):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.input_size=input_size

        self.i2h = nn.Linear(input_size+hidden_size,hidden_size)
        self.i2o = nn.Linear(input_size+hidden_size,output_size)
        self.softmax = nn.LogSoftmax(dim=1)
        self.hidden_tensor = self.init_hidden()
    def forward(self, target_state, tracker_state):
        x = torch.cat((target_state.squeeze(), tracker_state.squeeze()), dim=0).reshape(1,self.input_size)
        combined = torch.cat((x,self.hidden_tensor), dim=1)
        # hidden = self.init_hidden(1)  # Set batch size to 1
        self.hidden_tensor = self.i2h(combined)
        output = self.i2o(combined)
        output = self.softmax(output)
        return output.reshape(3,1)

    def init_hidden(self,batch_size=0):
        if batch_size==0:
            return torch.zeros(1,  self.hidden_size)

# # # Define the network dimensions
# input_size = 12
# hidden_size = 64
# output_size = 3
# # ### initial Tracker State ###
# tracker_state = torch.reshape(torch.tensor([[50.], [50.], [80.], [-3.], [4.], [4.]]), (6,1))
# #
# # ### Initial State 1ST and 2ND Moments ###
# m1x_0 = torch.reshape(torch.tensor([[10.], [10.], [90.], [-3.], [4.], [4.]]), (6,1))
# # # Create an instance of the ElmanRNN
# rnn = RNN(input_size, hidden_size, output_size)
# #
# # # Create a random input tensor
#
# #
# # # Forward pass
# output = rnn.forward(m1x_0,tracker_state)
# #
# print(output.shape)  # Should print torch.Size([ 3, 1])