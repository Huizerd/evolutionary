import torch
import torch.nn as nn
import torch.nn.functional as F


class ANN(nn.Module):
    def __init__(self, config):
        super(ANN, self).__init__()

        # Input layer size (related to encoding)
        self.encoding = config["net"]["encoding"]
        if self.encoding == "both":
            inputs = 2
        elif self.encoding == "divergence":
            inputs = 1
        else:
            raise ValueError("Invalid encoding")

        # Does Kaiming init for weights and biases,
        # while the paper inits the latter as zero
        # https://pouannes.github.io/blog/initialization/#mjx-eqn-eqfwd_K
        self.fc1 = nn.Linear(inputs, config["net"]["hidden size"])
        self.fc2 = nn.Linear(config["net"]["hidden size"], 1)

        # Turn off gradients
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        # x = self._encode(x)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

    def _encode(self, x):
        if self.encoding == "both":
            return x
        elif self.encoding == "divergence":
            return x[..., 0]

    def mutate(self, genes, types, mutation_rate=1.0):
        # Go over all genes that have to be mutated
        for gene in genes:
            for child in self.children():
                if (
                    hasattr(child, gene)
                    and (gene == "weight" or gene == "bias")
                    and "incremental" not in types
                ):
                    param = getattr(child, gene)
                    # Uniform in range [-w - 0.05, 2w + 0.05]
                    mutation = (3.0 * torch.rand_like(param) - 1.0) * param + (
                        2.0 * torch.rand_like(param) - 1.0
                    ) * 0.05
                    mask = torch.rand_like(param) < mutation_rate
                    param.masked_scatter_(mask, mutation)
                elif (
                    hasattr(child, gene)
                    and (gene == "weight" or gene == "bias")
                    and "incremental" in types
                ):
                    param = getattr(child, gene)
                    # Uniform increase/decrease from [-1, 1]
                    param += (torch.empty_like(param).uniform_(-1.0, 1.0)) * (
                        torch.rand_like(param) < mutation_rate
                    ).float()
                    param.clamp_(-3.0, 3.0)
