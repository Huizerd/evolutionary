import torch

from pysnn.network import SNNNetwork
from pysnn.connection import Linear
from pysnn.neuron import Input, AdaptiveLIFNeuron, LIFNeuron


class SNN(SNNNetwork):
    def __init__(self, config):
        super(SNN, self).__init__()

        # Get configuration parameters for connections and neurons
        # Trace parameters for input neurons don't matter, unused anyway
        n_in_dynamics = [
            config["net"]["dt"],
            config["net"]["alpha t"][0],
            config["net"]["tau t"][0],
        ]
        n_hid_dynamics = [
            config["net"]["thresh"][0],
            config["net"]["v rest"][0],
            config["net"]["alpha v"][0],
            config["net"]["alpha t"][0],
            config["net"]["dt"],
            config["net"]["refrac"][0],
            config["net"]["tau v"][0],
            config["net"]["tau t"][0],
            config["net"]["alpha thresh"][0],
            config["net"]["tau thresh"][0],
        ]
        n_out_dynamics = [
            config["net"]["thresh"][1],
            config["net"]["v rest"][1],
            config["net"]["alpha v"][1],
            config["net"]["alpha t"][1],
            config["net"]["dt"],
            config["net"]["refrac"][1],
            config["net"]["tau v"][1],
            config["net"]["tau t"][1],
            config["net"]["alpha thresh"][0],
            config["net"]["tau thresh"][0],
        ]
        c_dynamics = [1, config["net"]["dt"], config["net"]["delay"]]

        # Encoding/decoding
        self.encoding = config["net"]["encoding"]
        self.decoding = config["net"]["decoding"]
        self.out_scale = config["net"]["output scale"]
        self.out_offset = config["net"]["output offset"]
        self.in_bound = config["net"]["input bound"]
        self.in_scale = config["net"]["input scale"]
        self.in_size = config["net"]["input size"]
        self.in_sigma = (
            config["net"]["input bound"] * 2 / (config["net"]["input size"] - 1)
        )
        self.in_centers = (
            torch.pow(
                torch.linspace(
                    -config["net"]["input bound"],
                    config["net"]["input bound"],
                    config["net"]["input size"],
                ),
                3,
            )
            / (config["net"]["input bound"] ** 2)
        ).view(1, 1, config["net"]["input size"])
        self.out_bounds = config["env"]["g bounds"]

        # Input/output layer size (related to encoding/decoding)
        if self.encoding == "both":
            inputs = 4
        elif self.encoding == "divergence":
            inputs = 2
        elif self.encoding == "place":
            inputs = config["net"]["input size"]
        else:
            raise ValueError("Invalid encoding")
        if self.decoding == "single trace":
            outputs = 1
        elif self.decoding in ["max trace", "sum trace"]:
            outputs = 2
        elif self.decoding == "weighted trace":
            outputs = 5
        else:
            raise ValueError("Invalid decoding")

        # Neurons
        self.neuron0 = Input((1, 1, inputs), *n_in_dynamics)

        if config["net"]["neuron"][0] == "regular":
            self.neuron1 = LIFNeuron(
                (1, 1, config["net"]["hidden size"]), *n_hid_dynamics[:-2]
            )
        elif config["net"]["neuron"][0] == "adaptive":
            self.neuron1 = AdaptiveLIFNeuron(
                (1, 1, config["net"]["hidden size"]), *n_hid_dynamics
            )
        else:
            raise ValueError("Invalid neuron type for hidden layer")

        if config["net"]["neuron"][1] == "regular":
            self.neuron2 = LIFNeuron((1, 1, outputs), *n_out_dynamics[:-2])
        elif config["net"]["neuron"][1] == "adaptive":
            self.neuron2 = AdaptiveLIFNeuron((1, 1, outputs), *n_out_dynamics)
        else:
            raise ValueError("Invalid neuron type for output layer")

        # Connections
        self.fc1 = Linear(inputs, config["net"]["hidden size"], *c_dynamics)
        self.fc2 = Linear(config["net"]["hidden size"], outputs, *c_dynamics)

    def forward(self, x):
        # Input layer: encoding
        x = self._encode(x)
        x, trace = self.neuron0(x)  # same x as above (just fed through)

        # Hidden layer
        # So actually, divergence * weight is direct input current to neurons here
        # So I might as well not change anything about the parameters there!
        # Connection trace is not used (2nd argument)
        x, _ = self.fc1(x, trace)
        spikes, trace = self.neuron1(x)

        # Output layer
        x, _ = self.fc2(spikes, trace)
        _, trace = self.neuron2(x)

        return self._decode(trace)

    def mutate(self, genes, types, mutation_rate=1.0):
        # Go over all genes that have to be mutated
        for gene in genes:
            for child in self.children():
                if (
                    hasattr(child, gene)
                    and gene == "weight"
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
                    hasattr(child, gene) and gene == "weight" and "incremental" in types
                ):
                    param = getattr(child, gene)
                    # Uniform increase/decrease from [-1, 1]
                    param += (torch.empty_like(param).uniform_(-1.0, 1.0)) * (
                        torch.rand_like(param) < mutation_rate
                    ).float()
                    param.clamp_(-3.0, 3.0)
                elif (
                    hasattr(child, gene)
                    and gene in ["alpha_v", "alpha_t", "alpha_thresh"]
                    and "all" in types
                ):
                    param = getattr(child, gene)
                    param += (torch.empty_like(param).uniform_(-0.667, 0.667)) * (
                        torch.rand_like(param) < mutation_rate
                    ).float()
                    param.clamp_(0.0, 2.0)
                elif (
                    hasattr(child, gene)
                    and gene in ["alpha_v", "alpha_t", "alpha_thresh"]
                    and "all" not in types
                ):
                    param = getattr(child, gene)
                    if torch.rand(()) < mutation_rate:
                        # Same for all neurons in layer!
                        # Works because a sensible range for all alphas is [0, 2]
                        param.fill_(torch.rand(()) * 2.0)
                elif (
                    hasattr(child, gene)
                    and gene in ["tau_v", "tau_t", "tau_thresh"]
                    and "all" in types
                ):
                    param = getattr(child, gene)
                    param += (torch.empty_like(param).uniform_(-0.333, 0.333)) * (
                        torch.rand_like(param) < mutation_rate
                    ).float()
                    param.clamp_(0.0, 1.0)
                elif (
                    hasattr(child, gene)
                    and gene in ["tau_v", "tau_t", "tau_thresh"]
                    and "all" not in types
                ):
                    param = getattr(child, gene)
                    if torch.rand(()) < mutation_rate:
                        # Same for all neurons in layer!
                        # Works because all taus are [0, 1]
                        param.fill_(torch.rand(()))
                elif (
                    hasattr(child, gene)
                    and gene == "thresh"
                    and isinstance(child, LIFNeuron)
                ):
                    # Only mutate threshold for non-adaptive neuron
                    param = getattr(child, gene)
                    param += (torch.empty_like(param).uniform_(-0.333, 0.333)) * (
                        torch.rand_like(param) < mutation_rate
                    ).float()
                    param.clamp_(0.0, 1.0)

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
        elif self.encoding == "place":
            # Don't repeat here, but in place centers (more efficient)
            # Works, because we take only divergence, so input has shape (1, 1, 1) and
            # in_centers has shape (1, 1, centers)
            self.input = self.in_scale * torch.exp(
                -(input[..., 0].clamp_(-self.in_bound, self.in_bound) - self.in_centers)
                ** 2
                / (2.0 * self.in_sigma ** 2)
            )
            return self.input

    def _scale_output(self, output):
        return self.out_bounds[0] + (self.out_bounds[1] - self.out_bounds[0]) * (
            output / self.out_scale + self.out_offset
        )

    def _decode(self, out_trace):
        # Scale single trace
        if self.decoding == "single trace":
            trace = out_trace.view(-1)
            return self._scale_output(trace)
        # Maximum of two traces
        elif self.decoding == "max trace":
            trace = out_trace.view(-1)
            output = trace * torch.tensor(self.out_bounds)
            return output[trace.argmax()].view(-1)
        # Sum of two traces (one for positive, one for negative)
        elif self.decoding == "sum trace":
            trace = out_trace.view(-1)
            output = (trace - trace.flip(0)).abs() * torch.tensor(self.out_bounds)
            return output[trace.argmax()].view(-1)
        # Weighted average of five traces
        elif self.decoding == "weighted trace":
            trace = out_trace.view(-1)
            if trace.sum() > 0.0:
                output = (
                    trace * torch.linspace(self.out_bounds[0], -self.out_bounds[0], 5)
                ).sum() / trace.sum()
                return output.view(-1)
            else:
                return torch.tensor([0.0])
