import torch

from pysnn.network import SNNNetwork
from pysnn.connection import Linear
from pysnn.neuron import Input, AdaptiveLIFNeuron, LIFNeuron


class TwoLayerSNN(SNNNetwork):
    def __init__(self, config):
        super(TwoLayerSNN, self).__init__()

        # Check network sizes
        self._check_sizes(config["net"])

        # Get configuration parameters for connections and neurons
        # Parameters we evolve are set to 0; they will be randomized later
        # Except thresh for AdaptiveLIF, which is the value for resetting the adaptive threshold
        # dt, alpha_t, tau_t
        n_in_dynamics = [1, 0.0, 0.0]
        # thresh, v_rest, alpha_v, alpha_t, dt, refrac, tau_v, tau_t
        n_lif_dynamics = [0.0, 0.0, 0.0, 0.0, 1, 0, 0.0, 0.0]
        # thresh, v_rest, alpha_v, alpha_t, dt, refrac, tau_v, tau_t, alpha_thresh, tau_thresh
        n_alif_dynamics = [0.2, 0.0, 0.0, 0.0, 1, 0, 0.0, 0.0, 0.0, 0.0]
        # batch_size, dt, delay
        c_dynamics = [1, 1, 0]

        # Encoding
        self.encoding = config["net"]["encoding"]
        self.setpoint = config["evo"]["D setpoint"]

        # Decoding
        self.decoding = config["net"]["decoding"]
        self.out_bounds = config["env"]["g bounds"]

        # Neurons and connections
        self._build_network(
            config["net"], n_in_dynamics, n_lif_dynamics, n_alif_dynamics, c_dynamics
        )

        # Randomize initial parameters
        self._randomize_weights(-1.0, 1.0)
        self._randomize_neurons(config["evo"]["genes"])

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
        else:
            raise ValueError("Invalid encoding")

        # Check decoding
        if config["decoding"] == "weighted trace":
            pass
        else:
            raise ValueError("Invalid decoding")

    def _build_network(
        self, config, n_in_dynamics, n_lif_dynamics, n_alif_dynamics, c_dynamics
    ):
        # Input
        if config["neurons"][0] == "input":
            self.neuron0 = Input((1, 1, config["layer sizes"][0]), *n_in_dynamics)
        else:
            raise ValueError("Invalid neuron type for input layer")

        # Hidden
        if config["neurons"][1] == "regular":
            self.neuron1 = LIFNeuron((1, 1, config["layer sizes"][1]), *n_lif_dynamics)
        elif config["neurons"][1] == "adaptive":
            self.neuron1 = AdaptiveLIFNeuron(
                (1, 1, config["layer sizes"][1]), *n_alif_dynamics
            )
        else:
            raise ValueError("Invalid neuron type for hidden layer")

        # Output
        if config["neurons"][2] == "regular":
            self.neuron2 = LIFNeuron((1, 1, config["layer sizes"][2]), *n_lif_dynamics)
        elif config["neurons"][2] == "adaptive":
            self.neuron2 = AdaptiveLIFNeuron(
                (1, 1, config["layer sizes"][2]), *n_alif_dynamics
            )
        else:
            raise ValueError("Invalid neuron type for output layer")

        # Connections
        self.fc1 = Linear(
            config["layer sizes"][0], config["layer sizes"][1], *c_dynamics
        )
        self.fc2 = Linear(
            config["layer sizes"][1], config["layer sizes"][2], *c_dynamics
        )

    def forward(self, x):
        # Encoding
        x = self._encode(x)

        # Input layer
        x, trace = self.neuron0(x)

        # Hidden layer
        x, _ = self.fc1(x, trace)
        spikes, trace = self.neuron1(x)

        # Output layer
        x, _ = self.fc2(spikes, trace)
        _, trace = self.neuron2(x)

        # Decoding
        return self._decode(trace)

    def mutate(self, genes, mutation_rate=1.0):
        # Go over all genes that have to be mutated
        for gene in genes:
            for child in self.children():
                if hasattr(child, gene) and gene == "weight":
                    param = getattr(child, gene)
                    # Uniform increase/decrease from [-1, 1]
                    param += (torch.empty_like(param).uniform_(-1.0, 1.0)) * (
                        torch.rand_like(param) < mutation_rate
                    ).float()
                    param.clamp_(-3.0, 3.0)
                elif hasattr(child, gene) and gene in [
                    "alpha_v",
                    "alpha_t",
                    "alpha_thresh",
                ]:
                    param = getattr(child, gene)
                    param += (torch.empty_like(param).uniform_(-0.667, 0.667)) * (
                        torch.rand_like(param) < mutation_rate
                    ).float()
                    param.clamp_(0.0, 2.0)
                elif hasattr(child, gene) and gene in ["tau_v", "tau_t", "tau_thresh"]:
                    param = getattr(child, gene)
                    param += (torch.empty_like(param).uniform_(-0.333, 0.333)) * (
                        torch.rand_like(param) < mutation_rate
                    ).float()
                    param.clamp_(0.0, 1.0)
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

    def _randomize_weights(self, low, high):
        self.fc1.reset_weights(a=low, b=high)
        self.fc2.reset_weights(a=low, b=high)

    def _randomize_neurons(self, genes):
        # Go over all genes that have to be mutated
        for gene in genes:
            for child in self.children():
                if hasattr(child, gene) and gene in [
                    "alpha_v",
                    "alpha_t",
                    "alpha_thresh",
                ]:
                    param = getattr(child, gene)
                    param.uniform_(0.0, 2.0)
                elif hasattr(child, gene) and gene in ["tau_v", "tau_t", "tau_thresh"]:
                    param = getattr(child, gene)
                    param.uniform_(0.0, 1.0)
                elif (
                    hasattr(child, gene)
                    and gene == "thresh"
                    and isinstance(child, LIFNeuron)
                ):
                    # Only mutate threshold for non-adaptive neuron
                    param = getattr(child, gene)
                    param.uniform_(0.0, 1.0)

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

    def _decode(self, out_trace):
        # Weighted average of traces
        if self.decoding == "weighted trace":
            trace = out_trace.view(-1)
            if trace.sum() != 0.0:
                output = (
                    trace
                    * torch.linspace(
                        self.out_bounds[0], self.out_bounds[1], trace.shape[0]
                    )
                ).sum() / trace.sum()
                return output.view(-1)
            else:
                return torch.tensor([0.0])


class ThreeLayerSNN(TwoLayerSNN):
    def _build_network(
        self, config, n_in_dynamics, n_lif_dynamics, n_alif_dynamics, c_dynamics
    ):
        # Input
        if config["neurons"][0] == "input":
            self.neuron0 = Input((1, 1, config["layer sizes"][0]), *n_in_dynamics)
        else:
            raise ValueError("Invalid neuron type for input layer")

        # Hidden 1
        if config["neurons"][1] == "regular":
            self.neuron1 = LIFNeuron((1, 1, config["layer sizes"][1]), *n_lif_dynamics)
        elif config["neurons"][1] == "adaptive":
            self.neuron1 = AdaptiveLIFNeuron(
                (1, 1, config["layer sizes"][1]), *n_alif_dynamics
            )
        else:
            raise ValueError("Invalid neuron type for first hidden layer")

        # Hidden 2
        if config["neurons"][2] == "regular":
            self.neuron2 = LIFNeuron((1, 1, config["layer sizes"][2]), *n_lif_dynamics)
        elif config["neurons"][2] == "adaptive":
            self.neuron2 = AdaptiveLIFNeuron(
                (1, 1, config["layer sizes"][2]), *n_alif_dynamics
            )
        else:
            raise ValueError("Invalid neuron type for second hidden layer")

        # Output
        if config["neurons"][3] == "regular":
            self.neuron3 = LIFNeuron((1, 1, config["layer sizes"][3]), *n_lif_dynamics)
        elif config["neuron"][3] == "adaptive":
            self.neuron3 = AdaptiveLIFNeuron(
                (1, 1, config["layer sizes"][3]), *n_alif_dynamics
            )
        else:
            raise ValueError("Invalid neuron type for output layer")

        # Connections
        self.fc1 = Linear(
            config["layer sizes"][0], config["layer sizes"][1], *c_dynamics
        )
        self.fc2 = Linear(
            config["layer sizes"][1], config["layer sizes"][2], *c_dynamics
        )
        self.fc3 = Linear(
            config["layer sizes"][2], config["layer sizes"][3], *c_dynamics
        )

    def forward(self, x):
        # Encoding
        x = self._encode(x)

        # Input layer
        x, trace = self.neuron0(x)

        # First hidden layer
        x, _ = self.fc1(x, trace)
        spikes, trace = self.neuron1(x)

        # Second hidden layer
        x, _ = self.fc2(spikes, trace)
        spikes, trace = self.neuron2(x)

        # Output layer
        x, _ = self.fc3(spikes, trace)
        _, trace = self.neuron3(x)

        # Decoding
        return self._decode(trace)

    def _randomize_weights(self, low, high):
        self.fc1.reset_weights(a=low, b=high)
        self.fc2.reset_weights(a=low, b=high)
        self.fc3.reset_weights(a=low, b=high)
