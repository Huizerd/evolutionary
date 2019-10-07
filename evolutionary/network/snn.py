import torch

from pysnn.network import SNNNetwork
from pysnn.connection import Linear
from pysnn.neuron import InputTraceLinear, LIFNeuronTraceLinear


# TODO: is there even a time dim?
# TODO: do we ever use larger batch sizes?
# TODO: we may need longer than time to get a good max if we don't reset in between
# TODO: trace can of course accumulate over time, so do
# max_trace = (np.ones(time) * trace_decay ** np.arange(time - 1, -1, -1)).sum() * alpha_t


class SNN(SNNNetwork):
    def __init__(self, inputs, hidden, outputs, config):
        super(SNN, self).__init__()

        # Get configuration parameters for connections and neurons
        n_dynamics = [
            config["snn"]["thresh"],
            config["snn"]["v rest"],
            config["snn"]["alpha v"],
            config["snn"]["alpha t"],
            config["snn"]["dt"],
            config["snn"]["duration refrac"],
            config["snn"]["voltage decay"],
            config["snn"]["trace decay"],
        ]
        n_in_dynamics = [
            config["snn"]["dt"],
            config["snn"]["alpha t"],
            config["snn"]["trace decay"],
        ]
        c_dynamics = [
            config["snn"]["batch size"],
            config["snn"]["dt"],
            config["snn"]["delay"],
        ]

        # Encoding/decoding
        # TODO: is there another way to do this?
        self.scale = config["snn"]["scale"]
        self.div_clamp = (-config["snn"]["scale"], config["snn"]["scale"])
        self.offset = config["snn"]["offset"]
        self.max_trace = config["snn"]["max trace"]
        self.time = config["snn"]["time"]
        # TODO: make this nicer, nothing from env/G!
        self.output_bounds = [b * 9.81 for b in config["env"]["thrust bounds"]]
        self.method = config["snn"]["decoding"]

        # Connections
        self.fc1 = Linear(inputs, hidden, *c_dynamics)
        self.fc2 = Linear(hidden, outputs, *c_dynamics)

        # Neurons
        self.neuron0 = InputTraceLinear(
            (config["snn"]["batch size"], 1, inputs), *n_in_dynamics
        )
        self.neuron1 = LIFNeuronTraceLinear(
            (config["snn"]["batch size"], 1, hidden), *n_dynamics
        )
        self.neuron2 = LIFNeuronTraceLinear(
            (config["snn"]["batch size"], 1, outputs), *n_dynamics
        )

        # NOTE: usage of decays instead of taus implies linear neurons!
        assert (
            "voltage decay" in config["snn"]
            and isinstance(self.neuron0, InputTraceLinear)
            and isinstance(self.neuron1, LIFNeuronTraceLinear)
            and isinstance(self.neuron2, LIFNeuronTraceLinear)
        )

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
                mutation = (3.0 * torch.rand_like(param) - 1.0) * param.abs() + (
                    2.0 * torch.rand_like(param) - 1.0
                ) * 0.05
                param.data = torch.where(
                    torch.rand_like(param) < mutation_rate, mutation, param
                )
            elif "delay" in name:
                mutation = torch.randint_like(param, -1, 2)
                param += mutation
                param.clamp_(min=0)
            # elif "thresh" in name and not "center" in name:
            #     print("Thresholds found")
            # elif "decay" in name:
            #     print("Decay found")

    def _scale_input(self, input):
        return input / self.scale + self.offset

    def _encode(self, input):
        # TODO: a fold is missing in Input neuron, so doesn't make sense to pass spikes
        # TODO: create issue and for now use clamp + offset. Or use clamp anyway?
        # Clamp divergence to bounds to prevent negative firing rate
        input.clamp_(*self.div_clamp)
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
        if self.method == "ttfs":
            # TODO: this needs spike trains of time size, do we have those even?
            # TODO: currently just outputs minimum thrust
            # For now, this approach assumes two things:
            # - is computed on CPU (only then will it return the first occurrence)
            # - time is last dimension
            ttfs = out_spikes.argmax(-1).view(-1).float() / self.time
            return self._scale_output(ttfs)
        elif self.method == "trace":
            # TODO: is trace in PySNN a scalar value or a tensor of spikes?
            # TODO: would need max possible trace for some scaling
            # TODO: do we reset trace between passes? Or not and small sequences?
            trace = out_trace.view(-1) / self.max_trace
            return self._scale_output(trace)
        elif self.method == "psp":
            # TODO: neuron should be without spiking in this case? And quite some decay
            # TODO: isn't this practically equivalent to trace approach? Except different constants?
            raise NotImplementedError("PSP decoding hasn't been implemented yet!")
        else:
            raise KeyError("Not a valid method key!")
