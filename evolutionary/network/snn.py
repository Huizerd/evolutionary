import torch

from pysnn.network import SNNNetwork
from pysnn.connection import Linear
from pysnn.neuron import InputTraceLinear, LIFNeuronTraceLinear


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
        ]
        c_dynamics = [1, config["snn"]["dt"], config["snn"]["delay"]]

        # Encoding/decoding
        self.in_scale = config["snn"]["input scale"]
        self.in_offset = config["snn"]["input offset"]
        self.out_scale = config["snn"]["output scale"]
        self.output_bounds = [b * 9.81 for b in config["env"]["thrust bounds"]]
        self.decoding = config["snn"]["decoding"]

        # Neurons
        self.neuron0 = InputTraceLinear((1, 1, inputs), *n_in_dynamics)
        self.neuron1 = LIFNeuronTraceLinear((1, 1, hidden), *n_hid_dynamics)
        self.neuron2 = LIFNeuronTraceLinear((1, 1, outputs), *n_out_dynamics)

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

        return self._decode(spikes, trace)

    def mutate(self, genes, mutation_rate=1.0):
        # Go over all genes that have to be mutated
        for gene in genes:
            for child in self.children():
                if hasattr(child, gene):
                    if gene == "weight":
                        weight = getattr(child, gene)
                        # Uniform in range [-w - 0.05, 2w + 0.05]
                        mutation = (
                            3.0 * torch.rand_like(weight) - 1.0
                        ) * weight.abs() + (2.0 * torch.rand_like(weight) - 1.0) * 0.05
                        # .data is needed to access parameter
                        weight.data = torch.where(
                            torch.rand_like(weight) < mutation_rate, mutation, weight
                        )

    def _scale_input(self, input):
        return input / self.in_scale + self.in_offset

    def _encode(self, input):
        # Clamp divergence to bounds to prevent negative firing rate
        input.clamp_(-self.in_scale, self.in_scale)
        return self._scale_input(input)

    def _scale_output(self, output):
        return (
            self.output_bounds[0]
            + (self.output_bounds[1] - self.output_bounds[0]) * output
        )

    def _decode(self, out_spikes, out_trace):
        # What to use as decoding? Time to first spike, PSP, trace? We have trace anyway
        # Or do multiple options and let evolution decide?
        # Return 1d tensor!
        if self.decoding == "trace":
            trace = out_trace.view(-1) / self.out_scale
            return self._scale_output(trace)
        else:
            raise KeyError("Not a valid method key!")
