import torch
import torch.nn as nn
import torch.nn.functional as F


class ANN(nn.Module):
    def __init__(self, inputs, hidden, outputs):
        super(ANN, self).__init__()

        # Does Kaiming init for weights and biases,
        # while the paper inits the latter as zero
        # https://pouannes.github.io/blog/initialization/#mjx-eqn-eqfwd_K
        self.fc1 = nn.Linear(inputs, hidden)
        self.fc2 = nn.Linear(hidden, outputs)

        # Turn off gradients
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)

    def mutate(self, genes, mutation_rate=1.0):
        # Go over all genes that have to be mutated
        for gene in genes:
            for child in self.children():
                if hasattr(child, gene):
                    if gene == "weight" or gene == "bias":
                        param = getattr(child, gene)
                        # Uniform in range [-w - 0.05, 2w + 0.05]
                        mutation = (3.0 * torch.rand_like(param) - 1.0) * param + (
                            2.0 * torch.rand_like(param) - 1.0
                        ) * 0.05
                        # .data is needed to access parameter
                        param.data = torch.where(
                            torch.rand_like(param) < mutation_rate, mutation, param
                        )
