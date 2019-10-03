import torch
import numpy as np

from pysnn.network import SNNNetwork
from pysnn.connection import Linear
from pysnn.neuron import InputTraceLinear, LIFNeuronTraceLinear


# Connection and neuron dynamics as globals for now
# TODO: move this to some configuration file
# TODO: is there even a time dim?
time = 50
intensity = 128
n_samples = 20
# TODO: do we ever use larger batch sizes?
batch_size = 1
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
delay = 0

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
n_in_dynamics = (dt, alpha_t, trace_decay)
# c_dynamics = (batch_size, dt, delay, tau_t, alpha_t)
c_dynamics = (batch_size, dt, delay)

# Configuration for encoding
offset = 1.0
scale = 10.0
leak = 1.0
thresh_plus = 0.03
thresh_enc = 0.5
div_clamp = (-scale, scale)

# Configuration for decoding
method = "trace"
thrust = (-0.8 * 9.81, 0.5 * 9.81)
# TODO: we may need longer than time to get a good max if we don't reset in between
# max_trace = (np.ones(time) * trace_decay ** np.arange(time - 1, -1, -1)).sum() * alpha_t
max_trace = 1.0


class SNN(SNNNetwork):
    def __init__(self, inputs, hidden, outputs):
        super(SNN, self).__init__()

        # Connections
        self.fc1 = Linear(inputs, hidden, *c_dynamics)
        self.fc2 = Linear(hidden, outputs, *c_dynamics)

        # Neurons
        self.neuron0 = InputTraceLinear((batch_size, 1, inputs), *n_in_dynamics)
        self.neuron1 = LIFNeuronTraceLinear((batch_size, 1, hidden), *n_dynamics)
        self.neuron2 = LIFNeuronTraceLinear((batch_size, 1, outputs), *n_dynamics)

    def forward(self, input):
        # Input layer: encoding
        input = self._encode(input)
        x, t = self.neuron0(input)

        # Hidden layer
        x = self.fc1(x, t)
        spikes, t = self.neuron1(x)

        # Output layer
        x = self.fc2(spikes, t)
        spikes, t = self.neuron2(x)

        return self._decode(spikes, t)

    def mutate(self, mutation_rate=1.0):
        # Go over all parameters
        for name, param in self.named_parameters():
            if "weight" in name:
                # Uniform in range [-w - 0.05, 2w + 0.05]
                # No idea why this range
                # TODO: is this a correct/efficient way to replace all data in a tensor?
                mutation = (3.0 * torch.rand_like(param) - 1.0) * param + (
                    2.0 * torch.rand_like(param) - 1.0
                ) * 0.05
                param.data = torch.where(
                    torch.rand_like(param) < mutation_rate, mutation, param
                )

    def _encode(self, input):
        # TODO: a fold is missing in Input neuron, so doesn't make sense to pass spikes
        # TODO: create issue and for now use clamp + offset. Or use clamp anyway?
        # Clamp divergence to bounds to prevent negative firing rate
        input.clamp_(*div_clamp)
        # spikes = torch.zeros(*input.size(), time, dtype=torch.uint8)
        # current = input * scale + offset
        # voltage = torch.zeros(*input.size(), dtype=torch.float)
        # thresh = torch.ones(*input.size(), dtype=torch.float) * thresh_enc
        #
        # for i in range(time):
        #     voltage += current
        #     spikes[..., i] = voltage >= thresh
        #     voltage[spikes[..., i] == 1] = 0.0
        #     thresh[spikes[..., i] == 1] += thresh_plus
        #     voltage *= leak
        #
        # return spikes
        return input / scale + offset

    def _decode(self, out_spikes, out_trace):
        # What to use as decoding? Time to first spike, PSP, trace? We have trace anyway
        # Or do multiple options and let evolution decide?
        # Return 1d tensor!
        if method == "ttfs":
            # TODO: this needs spike trains of time size, do we have those even?
            # TODO: currently just outputs minimum thrust
            # For now, this approach assumes two things:
            # - is computed on CPU (only then will it return the first occurrence)
            # - time is last dimension
            ttfs = out_spikes.argmax(-1).view(-1).float() / time
            return thrust[0] + (thrust[1] - thrust[0]) * ttfs
        elif method == "trace":
            # TODO: is trace in PySNN a scalar value or a tensor of spikes?
            # TODO: would need max possible trace for some scaling
            # TODO: do we reset trace between passes? Or not and small sequences?
            return thrust[0] + (thrust[1] - thrust[0]) * (
                out_trace.view(-1) / max_trace
            )
        elif method == "psp":
            # TODO: neuron should be without spiking in this case? And quite some decay
            # TODO: isn't this practically equivalent to trace approach? Except different constants?
            raise NotImplementedError("PSP decoding hasn't been implemented yet!")
        else:
            raise KeyError("Not a valid method key!")
