import torch
import numpy as np

from pysnn.network import SNNNetwork
from pysnn.connection import Linear
from pysnn.neuron import LIFNeuronTraceLinear


# Connection and neuron dynamics as globals for now
# TODO: move this to some configuration file
time = 50
intensity = 128
n_samples = 20
batch_size = 20
thresh = 0.8
v_rest = 0
alpha_v = 0.2
tau_v = 5
alpha_t = 1.0
tau_t = 5
voltage_decay = 0.8  # taus are used for exponential neurons, decays for linear ones
trace_decay = 0.8
duration_refrac = 5
dt = 1
delay = 3

# n_dynamics = (thresh, v_rest, alpha_v, alpha_t, dt, duration_refrac, tau_v, tau_t)
n_dynamics = (
    thresh,
    v_rest,
    alpha_v,
    alpha_t,
    dt,
    duration_refrac,
    voltage_decay,
    trace_decay,
)
# c_dynamics = (batch_size, dt, delay, tau_t, alpha_t)
c_dynamics = (batch_size, dt, delay)

# Configuration for encoding
offset = 1.0
scale = 1.0
leak = 1.0
thresh_plus = 0.03
thresh_enc = 0.5

# Configuration for decoding
method = "ttfs"
thrust = (-0.8 * 9.81, 0.5 * 9.81)
# TODO: we may need longer than time to get a good max if we don't reset in between
max_trace = (np.ones(time) * trace_decay ** np.arange(time - 1, -1, -1)) * alpha_t


class SNN(SNNNetwork):
    def __init__(self, inputs, hidden, outputs):
        super(SNN, self).__init__()

        # Connections
        self.fc1 = Linear(inputs, hidden, *c_dynamics)
        self.fc2 = Linear(hidden, outputs, *c_dynamics)

        # Neurons
        self.neuron1 = LIFNeuronTraceLinear((batch_size, 1, hidden), *n_dynamics)
        self.neuron2 = LIFNeuronTraceLinear((batch_size, 1, outputs), *n_dynamics)

    def forward(self, input):
        # Input layer: encoding
        input = self._encode(input)

        # Hidden layer
        x, t = self.fc1(input)
        x = self.neuron1(x, t)

        # Output layer
        x, t = self.fc2(x)
        x = self.neuron2(x, t)

        return self._decode(x)

    def mutate(self, mutation_rate=1.0):
        # Go over all parameters
        for param in self.parameters():
            print(param.name)

    def _encode(self, input):
        spikes = torch.zeros(*input.size(), time, dtype=torch.uint8)
        current = input * scale + offset
        voltage = torch.zeros(*input.size(), dtype=torch.float)
        thresh = torch.ones(*input.size(), dtype=torch.float) * thresh_enc

        for i in range(time):
            voltage += current
            spikes[..., i] = voltage >= thresh
            voltage[spikes[..., i] == 1] = 0.0
            thresh[spikes[..., i] == 1] += thresh_plus
            voltage *= leak

        return spikes

    def _decode(self, output):
        # What to use as decoding? Time to first spike, PSP, trace? We have trace anyway
        # Or do multiple options and let evolution decide?
        # Return 1d tensor!
        if method == "ttfs":
            # For now, this approach assumes two things:
            # - is computed on CPU (only then will it return the first occurrence)
            # - time is last dimension
            ttfs = output.argmax(-1).squeeze().float() / time
            return thrust[0] + (thrust[1] - thrust[0]) * ttfs
        elif method == "trace":
            # TODO: is trace in PySNN a scalar value or a tensor of spikes?
            # TODO: would need max possible trace for some scaling
            # TODO: do we reset trace between passes? Or not and small sequences?
            return thrust[0] + (thrust[1] - thrust[0]) * (
                self.neuron2.trace / max_trace
            )
        elif method == "psp":
            # TODO: neuron should be without spiking in this case? And quite some decay
            # TODO: isn't this practically equivalent to trace approach? Except different constants?
            raise NotImplementedError("PSP decoding hasn't been implemented yet!")
        else:
            raise KeyError("Not a valid method key!")
