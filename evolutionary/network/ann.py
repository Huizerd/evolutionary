import numpy as np

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

    def mutate(self, mutation_rate=1.0):
        # Go over all weights and biases
        for param in self.parameters():
            # Create mask for mutation rate

            # Uniform in range [-w - 0.05, 2w + 0.05]
            # No idea why this range
            # TODO: is this a correct/efficient way to replace all data in a tensor?
            mutation = (3.0 * torch.rand_like(param) - 1.0) * param + (
                2.0 * torch.rand_like(param) - 1.0
            ) * 0.05
            param.data = torch.where(
                torch.rand_like(param) < mutation_rate, mutation, param
            )


class ANNkirk:
    def __init__(self, inputs, hidden, outputs):
        # Needed elsewhere
        self.nn_shape = (inputs, hidden, outputs)

        # Initialize weights and biases
        self.weights = np.array(
            [
                np.zeros([self.nn_shape[layer], self.nn_shape[layer + 1]])
                for layer, _ in enumerate(self.nn_shape[:-1])
            ]
        )
        self.bias = np.array([np.zeros(neurons) for neurons in self.nn_shape])

        for layer_nr, layer_shape in enumerate(self.nn_shape[:-1]):
            self.weights[layer_nr] = self._init_rand(
                layer_shape, self.nn_shape[layer_nr + 1]
            )
        for layer_nr, layer_shape in enumerate(self.nn_shape):
            self.bias[layer_nr] = self._init_rand(layer_shape)

    def _activation(self, x):
        return np.maximum(x, 0.0)

    def _init_rand(self, *args, **kwargs):
        # Initialize randomly to range [-0.2, 0.2]
        return (np.random.rand(*args, **kwargs) - 0.5) / 2.5

    def forward(self, x):
        # Initialize and set input activation
        activation = np.array(
            [np.zeros(self.nn_shape[layer]) for layer, _ in enumerate(self.nn_shape)]
        )
        activation[0][:] = np.array(x[: self.nn_shape[0]]) + self.bias[0]

        # Linear layer
        for layer_nr, weights_layer in enumerate(self.weights):
            activation[layer_nr + 1][:] = (
                activation[layer_nr].dot(weights_layer) + self.bias[layer_nr + 1]
            )

            # Apply activation function for all but last layer
            if layer_nr + 1 < len(self.weights):
                activation[layer_nr + 1][:] = self._activation(activation[layer_nr + 1])

        return activation[-1]

    def _perturb(self, x):
        # Generate random value in range [-w - 0.05, 2w + 0.05]
        return (3.0 * np.random.random() - 1.0) * x + (
            2.0 * np.random.random() - 1.0
        ) * 0.05

    def mutate(self, mutation_rate=1.0):
        for layer_nr, s in enumerate(self.nn_shape):
            for neuron in range(s):
                if np.random.random() <= mutation_rate:
                    self.bias[layer_nr][neuron] = self._perturb(
                        self.bias[layer_nr][neuron]
                    )
                if layer_nr + 1 < len(self.nn_shape):
                    for connection in range(self.nn_shape[layer_nr + 1]):
                        if np.random.random() <= mutation_rate:
                            self.weights[layer_nr][neuron][connection] = self._perturb(
                                self.weights[layer_nr][neuron][connection]
                            )
