
import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import math
class ElmanTransformer(nn.Module):
    def __init__(self, input_size=12, hidden_size=64, output_size=3, num_layers=2, num_heads=4):
        super(ElmanTransformer, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Linear(input_size, hidden_size)
        self.positional_encoding = PositionalEncoding(hidden_size)
        encoder_layers = TransformerEncoderLayer(hidden_size, num_heads)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers)
        self.fc = nn.Linear(hidden_size, output_size)
        self.input_size = input_size
    def forward(self, target_state, tracker_state):
        x = torch.cat((target_state.squeeze(), tracker_state.squeeze()), dim=0).reshape(1,1,self.input_size)
        x = self.embedding(x)
        x = self.positional_encoding(x)
        output = self.transformer_encoder(x)
        output = output.squeeze(1)
        output = self.fc(output)
        return output.reshape(3,1)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return x
# # Define the network dimensions
# input_size = 12
# hidden_size = 64
# output_size = 3
# num_layers = 2
# num_heads = 4
# #
# # Initial Tracker State
# tracker_state = torch.reshape(torch.tensor([[50.], [50.], [80.], [-3.], [4.], [4.]]), (6, 1))
#
# # Initial State 1st and 2nd Moments
# m1x_0 = torch.reshape(torch.tensor([[10.], [10.], [90.], [-3.], [4.], [4.]]), (6, 1))
#
# # Create an instance of the ElmanLSTM
# elman_transformer = ElmanTransformer(input_size, hidden_size, output_size)
#
# # Create a random input tensor
# input_tensor = torch.randn(12, 1)
#
# # Forward pass
# output = elman_transformer.forward(m1x_0, tracker_state)
#
# print(output.shape)  # Should print torch.Size([1, 3])