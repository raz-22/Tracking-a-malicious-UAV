import torch
import torch.nn as nn
import torch.nn.init as init

class MLP(nn.Module):
    def __init__(self, hidden_size=64):
        super(MLP, self).__init__()

        self.hidden_size = hidden_size

        # Define the two hidden layers
        self.fc1 = nn.Linear(12, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)

        # Define the output layer
        self.fc3 = nn.Linear(hidden_size, 3)
    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
    def forward(self, x):
        # Pass the input through the hidden layers with ReLU activation
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))

        # Pass the output through the output layer with linear activation
        x = self.fc3(x)

        return x


"""
  
  
    def train(model, dataloader, criterion, optimizer, num_epochs=10):
        # Define the optimizer and compile the model
        optimizer = torch.optim.Adam(model.parameters())
        criterion = information_theoretic_cost
        for epoch in range(num_epochs):
            running_loss = 0.0
            model.train()
            # Iterate over the dataloader
            for inputs in dataloader:
                # Zero the gradients
                optimizer.zero_grad()

                # Compute the output of the model
                outputs = model(inputs)

                # Compute the loss using the information theoretic cost function
                loss = criterion(outputs, inputs)  # assuming inputs contain the target state

                # Backpropagate the gradients and update the model parameters
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            # Print the average loss for the epoch
            print("Epoch %d loss: %.3f" % (epoch + 1, running_loss / len(dataloader)))

"""