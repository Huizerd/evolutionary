import torch.nn as nn
import torch.fucntional as F


class ANN(nn.Module):
    def __init(self, inputs, hidden, outputs):
        super(ANN, self).__init__()

        # TODO: again check weight init
        self.fc1 = nn.Linear(inputs, hidden)
        self.fc2 = nn.Linear(hidden, outputs)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)

    def mutate(self, mutation_rate=1.0):
        # TODO: implement mutations
        pass
