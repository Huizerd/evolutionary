import torch
import torch.nn as nn
import torch.nn.functional as F


class ANN(nn.Module):
    def __init__(self, inputs, hidden, outputs):
        super(ANN, self).__init__()

        # TODO: again check weight init
        self.fc1 = nn.Linear(inputs, hidden)
        self.fc2 = nn.Linear(hidden, outputs)

        # Turn off gradients
        # TODO: is this the way to do this?
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)

    def mutate(self, mutation_rate=1.0):
        # Go over all weights and biases
        for param in self.parameters():
            # Create mask for mutation rate
            # mask = torch.rand_like(param) < mutation_rate

            # Uniform in range [-w - 0.05, 2w + 0.05]
            # No idea why this range
            # TODO: is this a correct/efficient way to replace all data in a tensor?
            mutation = (3.0 * torch.rand_like(param) - 1.0) * param + (
                2.0 * torch.rand_like(param) - 1.0
            ) * 0.05
            # TODO: use torch.where, not +=!
            # TODO: or formulate this as a change to the weight instead of replacing it?
            # TODO: data to actually access it?
            param.data = torch.where(
                torch.rand_like(param) < mutation_rate, mutation, param
            )
            # param += mask.float() * mutation
