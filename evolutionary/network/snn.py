import torch

from pysnn.network import SNNNetwork
from pysnn.connection import Linear
from pysnn.neuron import Input, AdaptiveLIFNeuron, LIFNeuron


class SNN(SNNNetwork):
    def __init__(self, inputs, hidden, outputs, config):
        super(SNN, self).__init__()

        # Get configuration parameters for connections and neurons
        # Trace parameters for input neurons don't matter, unused anyway
        n_in_dynamics = [
            config["snn"]["dt"],
            config["snn"]["alpha t"][0],
            config["snn"]["tau t"][0],
        ]
        n_hid_dynamics = [
            config["snn"]["thresh"][0],
            config["snn"]["v rest"][0],
            config["snn"]["alpha v"][0],
            config["snn"]["alpha t"][0],
            config["snn"]["dt"],
            config["snn"]["refrac"][0],
            config["snn"]["tau v"][0],
            config["snn"]["tau t"][0],
            config["snn"]["alpha thresh"][0],
            config["snn"]["tau thresh"][0],
        ]
        n_out_dynamics = [
            config["snn"]["thresh"][1],
            config["snn"]["v rest"][1],
            config["snn"]["alpha v"][1],
            config["snn"]["alpha t"][1],
            config["snn"]["dt"],
            config["snn"]["refrac"][1],
            config["snn"]["tau v"][1],
            config["snn"]["tau t"][1],
            config["snn"]["alpha thresh"][0],
            config["snn"]["tau thresh"][0],
        ]
        c_dynamics = [1, config["snn"]["dt"], config["snn"]["delay"]]

        # Encoding/decoding
        self.double_neurons = config["double neurons"]
        self.in_scale = config["snn"]["input scale"]
        self.in_offset = config["snn"]["input offset"]
        self.double_actions = config["double actions"]
        self.out_scale = config["snn"]["output scale"]
        self.out_offset = config["snn"]["output offset"]
        self.output_bounds = [b * 9.81 for b in config["env"]["thrust bounds"]]
        self.decoding = config["snn"]["decoding"]
        if self.decoding == "potential":
            n_out_dynamics[0] = float("inf")

        # Neurons
        self.neuron0 = Input((1, 1, inputs), *n_in_dynamics)
        self.neuron1 = AdaptiveLIFNeuron((1, 1, hidden), *n_hid_dynamics)
        self.neuron2 = AdaptiveLIFNeuron((1, 1, outputs), *n_out_dynamics)

        # Connections
        self.fc1 = Linear(inputs, hidden, *c_dynamics)
        self.fc2 = Linear(hidden, outputs, *c_dynamics)

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
        spikes, trace = self.neuron2(x)

        return self._decode(spikes, trace, self.neuron2.v_cell)

    def mutate(self, genes, mutation_rate=1.0):
        # Go over all genes that have to be mutated
        for gene in genes:
            for child in self.children():
                if hasattr(child, gene):
                    if gene == "weight":
                        weight = getattr(child, gene)
                        # Uniform in range [-w - 0.05, 2w + 0.05]
                        mutation = (3.0 * torch.rand_like(weight) - 1.0) * weight + (
                            2.0 * torch.rand_like(weight) - 1.0
                        ) * 0.05
                        # .data is needed to access parameter
                        weight.data = torch.where(
                            torch.rand_like(weight) < mutation_rate, mutation, weight
                        )

    def _scale_input(self, input):
        return input / self.in_scale + self.in_offset

    def _encode(self, input):
        if self.double_neurons:
            # Repeat to have: (div, divdot, div, divdot)
            self.input = input.repeat(1, 1, 2)
            # Clamp first half to positive, second half to negative
            self.input[..., :2].clamp_(min=0)
            self.input[..., 2:].clamp_(max=0)
            # TODO: but when divergence is now zero, our action will go to zero as well!
            # TODO: so use offset to guarantee firing?
            # TODO: or decode action with an offset = thrust for hover?
            # TODO: or two output neurons as well??
            # TODO: no need for maximum clamping, because neuron saturation takes care of that right?
            return self.input.abs()
        else:
            # Clamp divergence to bounds to prevent negative firing rate
            input.clamp_(-self.in_scale, self.in_scale)
            return self._scale_input(input)

    def _scale_output(self, output):
        return self.output_bounds[0] + (
            self.output_bounds[1] - self.output_bounds[0]
        ) * (output / self.out_scale + self.out_offset)

    def _decode(self, out_spikes, out_trace, out_volt):
        # What to use as decoding? Time to first spike, PSP, trace? We have trace anyway
        # Or do multiple options and let evolution decide?
        # Return 1d tensor!
        if self.double_actions:
            if self.decoding == "max trace":
                trace = out_trace.view(-1)
                output = trace * torch.tensor(self.output_bounds)
                return output[trace.argmax()].view(-1)
            elif self.decoding == "sum trace":
                trace = out_trace.view(-1)
                output = (trace - trace.flip(0)).abs() * torch.tensor(
                    self.output_bounds
                )
                return output[trace.argmax()].view(-1)
            elif self.decoding == "potential":
                volt = out_volt.view(-1)
                output = (volt - volt.flip(0)).abs() * torch.tensor(self.output_bounds)
                return output[volt.argmax()].view(-1)
            else:
                raise KeyError("Not a valid method key!")
        else:
            if self.decoding == "trace":
                trace = out_trace.view(-1)
                return self._scale_output(trace)
            elif self.decoding == "potential":
                volt = out_volt.view(-1)
                output = volt.abs() * torch.tensor(self.output_bounds)
                if volt > 0:
                    return output[1].view(-1)
                else:
                    return output[0].view(-1)
            else:
                raise KeyError("Not a valid method key!")
