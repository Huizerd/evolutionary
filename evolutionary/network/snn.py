from collections import deque

import torch
import numpy as np

from pysnn.network import SNNNetwork
from pysnn.connection import Linear
from pysnn.neuron import Input, LIFNeuron


class CustomLIFNeuron(LIFNeuron):
    def update_voltage(self, x):
        self.v_cell = self.voltage_update(
            self.v_cell,
            self.v_rest,
            x,
            self.alpha_v,
            (4096 - self.tau_v) / 4096,
            self.dt,
            self.refrac_counts,
        )


class TwoLayerSNN(SNNNetwork):
    def __init__(self, config):
        super(TwoLayerSNN, self).__init__()

        # Check network sizes
        self._check_sizes(config["net"])

        # Get configuration parameters for connections and neurons
        # Parameters we evolve are set to 0; they will be randomized later
        # dt, alpha_t, tau_t
        n_in_dynamics = [1, 0.0, 0.0]
        # thresh, v_rest, alpha_v, alpha_t, dt, refrac, tau_v, tau_t
        n_lif_dynamics = [0.0, 0.0, 1.0, 0.0, 1, 0, 0.0, 0.0]
        # batch_size, dt, delay
        c_dynamics = [1, 1, 0]

        # Weight exponent
        self.weight_exp = config["net"]["weight exp"]

        # Encoding
        self.encoding = config["net"]["encoding"]
        self.setpoint = config["evo"]["D setpoint"]
        # One less step to also allow coverage for values outside the range
        self.encs = config["net"]["layer sizes"][0]
        self.buckets = torch.linspace(
            -10.0, 10.0, steps=config["net"]["layer sizes"][0] - 1
        )
        if config["net"]["encoding"] == "cubed-spike place":
            self.buckets = torch.pow(self.buckets, 3) / (10 ** 2)

        # Decoding
        self.decoding = config["net"]["decoding"]
        self.out_bounds = config["env"]["g bounds"]
        self.trace_weights = torch.linspace(
            self.out_bounds[0], self.out_bounds[1], config["net"]["layer sizes"][-1]
        )

        # Neurons and connections
        self._build_network(config["net"], n_in_dynamics, n_lif_dynamics, c_dynamics)

        # Randomize initial parameters
        self._randomize_weights(
            *config["evo"]["limits"][config["evo"]["genes"].index("weight")]
        )
        self._randomize_neurons(config["evo"]["genes"], config["evo"]["limits"])

    def _check_sizes(self, config):
        # Check encoding and input size
        if config["encoding"] == "both" or config["encoding"] == "both setpoint":
            assert (
                config["layer sizes"][0] == 4
            ), "'both'/'both setpoint' encodings needs input size of 4"
        elif config["encoding"] == "divergence":
            assert (
                config["layer sizes"][0] == 2
            ), "'divergence' encoding needs input size of 2"
        elif (
            config["encoding"] == "single-spike place"
            or config["encoding"] == "cubed-spike place"
        ):
            assert (
                config["layer sizes"][0] % 2 == 0
            ), "It makes sense to have an uneven number of bins (so even number of neurons to account for both extremes), such that there is a bound on 0"
        elif config["encoding"] == "double float16":
            assert (
                config["layer sizes"][0] == 32
            ), "'divergence' encoding needs input size of 32"
        else:
            raise ValueError("Invalid encoding")

        # Check decoding
        if config["decoding"] == "weighted trace":
            pass
        else:
            raise ValueError("Invalid decoding")

    def _build_network(self, config, n_in_dynamics, n_lif_dynamics, c_dynamics):
        # Input
        if config["neurons"][0] == "input":
            self.neuron0 = Input((1, 1, config["layer sizes"][0]), *n_in_dynamics)
        else:
            raise ValueError("Invalid neuron type for input layer")

        # Input buffer
        self.in_buffer = deque(
            [
                torch.zeros_like(self.neuron0.trace)
                for _ in range(max(1, config["buffer"]))
            ],
            maxlen=max(1, config["buffer"]),
        )

        # Hidden
        if config["neurons"][1] == "regular":
            self.neuron1 = CustomLIFNeuron(
                (1, 1, config["layer sizes"][1]), *n_lif_dynamics
            )
        else:
            raise ValueError("Invalid neuron type for hidden layer")

        # Hidden buffer
        self.hid_buffer = deque(
            [
                [
                    torch.zeros_like(self.neuron0.trace),
                    torch.zeros_like(self.neuron0.trace),
                ]
                for _ in range(max(1, config["buffer"]))
            ],
            maxlen=max(1, config["buffer"]),
        )

        # Output
        if config["neurons"][2] == "regular":
            self.neuron2 = CustomLIFNeuron(
                (1, 1, config["layer sizes"][2]), *n_lif_dynamics
            )
        else:
            raise ValueError("Invalid neuron type for output layer")
        # Name of output neuron for trace mutation
        self.out_name = "neuron2"

        # Output buffer
        self.out_buffer = deque(
            [
                [
                    torch.zeros_like(self.neuron1.trace),
                    torch.zeros_like(self.neuron1.trace),
                ]
                for _ in range(max(1, config["buffer"]))
            ],
            maxlen=max(1, config["buffer"]),
        )

        # Connections
        self.fc1 = Linear(
            config["layer sizes"][0], config["layer sizes"][1], *c_dynamics
        )
        self.fc2 = Linear(
            config["layer sizes"][1], config["layer sizes"][2], *c_dynamics
        )

    def reset_state(self):
        super(TwoLayerSNN, self).reset_state()

        # Reset buffers
        self.in_buffer = deque(
            [torch.zeros_like(self.neuron0.trace) for _ in range(len(self.in_buffer))],
            maxlen=len(self.in_buffer),
        )
        self.hid_buffer = deque(
            [
                [
                    torch.zeros_like(self.neuron0.trace),
                    torch.zeros_like(self.neuron0.trace),
                ]
                for _ in range(len(self.hid_buffer))
            ],
            maxlen=len(self.hid_buffer),
        )
        self.out_buffer = deque(
            [
                [
                    torch.zeros_like(self.neuron1.trace),
                    torch.zeros_like(self.neuron1.trace),
                ]
                for _ in range(len(self.out_buffer))
            ],
            maxlen=len(self.out_buffer),
        )

    def forward(self, x):
        # Encoding
        x = self._encode(x)

        # Input layer
        self.in_buffer.append(x)
        x, trace = self.neuron0(self.in_buffer[0])

        # Hidden layer
        self.hid_buffer.append([x, trace])
        x, _ = self.fc1(self.hid_buffer[0][0], self.hid_buffer[0][1])
        spikes, trace = self.neuron1(x * 2 ** self.weight_exp)
        self.hid_spikes = spikes.view(-1)

        # Output layer
        self.out_buffer.append([spikes, trace])
        x, _ = self.fc2(self.out_buffer[0][0], self.out_buffer[0][1])
        spikes, trace = self.neuron2(x * 2 ** self.weight_exp)

        # Decoding
        return self._decode(spikes, trace)

    def mutate(self, genes, mutations, limits, decay, mutation_rate=1.0):
        # Go over all genes that have to be mutated
        for gene, mut, lim in zip(genes, mutations, limits):
            for name, child in self.named_children():
                if hasattr(child, gene) and gene == "weight":
                    param = getattr(child, gene)
                    param += (
                        torch.randint_like(
                            param, round(mut[0] * decay), round(mut[1] * decay) + 1
                        )
                        * (torch.rand_like(param) < mutation_rate).float()
                    )
                    param.clamp_(*lim)
                    param += param % 2
                # Only trace of output neuron is used
                elif (
                    hasattr(child, gene)
                    and gene in ["alpha_t", "tau_t"]
                    and name == self.out_name
                ):
                    param = getattr(child, gene)
                    param += (
                        torch.empty_like(param).uniform_(*mut)
                        * (torch.rand_like(param) < mutation_rate).float()
                        * decay
                    )
                    param.clamp_(*lim)
                elif hasattr(child, gene) and gene in ["tau_v", "thresh"]:
                    param = getattr(child, gene)
                    param += (
                        torch.randint_like(
                            param, round(mut[0] * decay), round(mut[1] * decay) + 1
                        )
                        * (torch.rand_like(param) < mutation_rate).float()
                    )
                    param.clamp_(*lim)

    def _randomize_weights(self, low, high):
        # Init
        self.fc1.weight.data = torch.randint_like(self.fc1.weight.data, low, high + 1)
        self.fc2.weight.data = torch.randint_like(self.fc2.weight.data, low, high + 1)
        # And round to even numbers only
        # Bounds are even, so no need for clamp!
        self.fc1.weight.data += self.fc1.weight.data % 2
        self.fc2.weight.data += self.fc2.weight.data % 2

    def _randomize_neurons(self, genes, limits):
        # Go over all genes that have to be mutated
        for gene, lim in zip(genes, limits):
            for name, child in self.named_children():
                # Only trace of output neuron is used
                if (
                    hasattr(child, gene)
                    and gene in ["alpha_t", "tau_t"]
                    and name == self.out_name
                ):
                    param = getattr(child, gene)
                    param.uniform_(*lim)
                elif hasattr(child, gene) and gene in ["tau_v", "thresh"]:
                    param = getattr(child, gene)
                    # .data is needed here
                    param.data = torch.randint_like(param.data, lim[0], lim[1] + 1)

    def _encode(self, input):
        if self.encoding == "both":
            # Repeat to have: (div, divdot, div, divdot)
            self.input = input.repeat(1, 1, 2)
            # Clamp first half to positive, second half to negative
            self.input[..., :2].clamp_(min=0.0)
            self.input[..., 2:].clamp_(max=0.0)
            return self.input.abs()

        elif self.encoding == "divergence":
            # Repeat to have: (div, div)
            self.input = input[..., 0].repeat(1, 1, 2)
            # Clamp first half to positive, second half to negative
            self.input[..., :1].clamp_(min=0.0)
            self.input[..., 1:].clamp_(max=0.0)
            return self.input.abs()

        elif self.encoding == "both setpoint":
            # Subtract setpoint
            input[..., 0] -= self.setpoint
            # Repeat to have: (div, divdot, div, divdot)
            self.input = input.repeat(1, 1, 2)
            # Clamp first half to positive, second half to negative
            self.input[..., :2].clamp_(min=0.0)
            self.input[..., 2:].clamp_(max=0.0)
            return self.input.abs()

        elif (
            self.encoding == "single-spike place"
            or self.encoding == "cubed-spike place"
        ):
            # Subtract setpoint
            input[..., 0] -= self.setpoint
            # Single spike based on D value in range [-10, 10]
            spike = torch.bucketize(input[..., 0], self.buckets)
            self.input = torch.zeros(1, 1, self.encs)
            self.input[..., spike] = 1
            return self.input

        elif self.encoding == "double float16":
            # Subtract setpoint
            input[..., 0] -= self.setpoint
            # Encode both D and Ddot as 16-bit float in binary
            # Subtract setpoint
            Dbin = bin(np.float16(input[..., 0].item()).view("H"))[2:].zfill(16)
            Ddotbin = bin(np.float16(input[..., 1].item()).view("H"))[2:].zfill(16)
            self.input = input.repeat(1, 1, 16)
            self.input[..., :16] = torch.from_numpy(np.array(list(Dbin)).astype(float))
            self.input[..., 16:] = torch.from_numpy(
                np.array(list(Ddotbin)).astype(float)
            )
            return self.input

    def _decode(self, out_spikes, out_trace):
        # Weighted average of traces
        if self.decoding == "weighted trace":
            self.out_spikes = out_spikes.view(-1)
            self.out_trace = out_trace.view(-1)
            if self.out_trace.sum() != 0.0:
                output = (
                    self.out_trace * self.trace_weights
                ).sum() / self.out_trace.sum()
                return output.view(-1)
            else:
                return torch.tensor([0.0])
