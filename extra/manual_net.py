from collections import OrderedDict

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from pysnn.network import SNNNetwork
from pysnn.connection import Linear
from pysnn.neuron import Input, Neuron, InputTraceLinear, LIFNeuronTraceLinear

from gym_quad.envs import QuadHover


class SNN(SNNNetwork):
    def __init__(
        self,
        inputs,
        hidden,
        outputs,
        nin_param,
        nhid_param,
        nout_param,
        c_param,
        enc_param,
        dec_param,
    ):
        super(SNN, self).__init__()

        # Neurons
        self.neuron0 = InputTraceLinear((1, 1, inputs), *nin_param)
        # self.neuron0 = LIFNeuronTraceLinear((1, 1, inputs), *nin_param)
        self.neuron1 = LIFNeuronTraceLinear((1, 1, hidden), *nhid_param)
        self.neuron2 = LIFNeuronTraceLinear((1, 1, outputs), *nout_param)

        # Connections
        self.fc1 = Linear(inputs, hidden, *c_param)
        self.fc2 = Linear(hidden, outputs, *c_param)

        # Encoding
        self.in_scale, self.in_offset = enc_param

        # Decoding
        self.out_bounds = dec_param

    def forward(self, x):
        # Input layer: encoding
        x = self._encode(x)
        spikes, trace = self.neuron0(x)

        # Hidden layer
        x, _ = self.fc1(spikes, trace)
        spikes, trace = self.neuron1(x)

        # Output layer
        x, _ = self.fc2(spikes, trace)
        spikes, trace = self.neuron2(x)

        return self._decode(spikes, trace)

    def _scale_input(self, input):
        return input / self.in_scale + self.in_offset

    def _encode(self, input):
        input.clamp_(-self.in_scale, self.in_scale)
        return self._scale_input(input)

    def _scale_output(self, output):
        return self.out_bounds[0] + (self.out_bounds[1] - self.out_bounds[0]) * output

    def _decode(self, out_spikes, out_trace):
        trace = out_trace.view(-1)
        return self._scale_output(trace)


if __name__ == "__main__":
    # Env parameters
    delay_env = (3, 6)
    noise = (0.1, 0.15)
    noise_p = (0.1, 0.25)
    thrust_bounds = (-0.8, 0.5)
    grav = 9.81
    thrust_tc = (0.02, 0.1)
    settle = 1.0
    wind = 0.1
    h0 = [2.0, 4.0, 6.0, 8.0]
    dt_env = 0.02
    seeds = 100

    # Create env
    env = QuadHover(
        delay=np.random.randint(*delay_env),
        comp_delay_prob=0.0,
        noise=np.random.uniform(*noise),
        noise_p=np.random.uniform(*noise_p),
        thrust_bounds=thrust_bounds,
        thrust_tc=np.random.uniform(*thrust_tc),
        settle=settle,
        wind=wind,
        h0=h0[0],
        dt=dt_env,
        seed=np.random.randint(seeds),
    )

    # SNN parameters
    # Constant
    dt_snn = 1
    delay_snn = 0
    v_rest = 0.0
    thresh = 1.0
    refrac = 0
    in_scale = 10.0
    in_offset = 1.0
    out_bounds = [b * grav for b in thrust_bounds]

    # Tunable
    hidden = 1
    alpha_t = [0.4, 0.4, 0.06165]
    alpha_v = [0.2, 0.2]
    t_decay = [0.9, 0.9, 0.95]
    v_decay = [0.8, 0.8]

    # Together in lists
    nin_param = [dt_snn, alpha_t[0], t_decay[0]]
    nhid_param = [
        thresh,
        v_rest,
        alpha_v[0],
        alpha_t[1],
        dt_snn,
        refrac,
        v_decay[0],
        t_decay[1],
    ]
    nout_param = [
        thresh,
        v_rest,
        alpha_v[1],
        alpha_t[2],
        dt_snn,
        refrac,
        v_decay[1],
        t_decay[2],
    ]
    c_param = [1, dt_snn, delay_snn]
    enc_param = [in_scale, in_offset]
    dec_param = out_bounds

    # Build SNN
    network = SNN(
        2, hidden, 1, nin_param, nhid_param, nout_param, c_param, enc_param, dec_param
    )

    # Adjust weights
    network.fc1.weight.data = torch.tensor([10.0, 0.0]).view(1, -1)
    network.fc2.weight.data = torch.tensor([10.0]).view(1, -1)

    # Go over all heights
    for h in h0:
        # Reset network and env
        network.reset_state()
        obs = env.reset(h0=h)
        done = False

        # For plotting
        state_list = []
        obs_gt_list = []
        obs_list = []
        time_list = []

        # For neuron visualization
        neuron_dict = OrderedDict(
            [
                (name, {"trace": [], "volt": [], "spike": [], "thresh": []})
                for name, child in network.named_children()
                if isinstance(child, Input) or isinstance(child, Neuron)
            ]
        )

        while not done:
            # Log performance
            state_list.append(env.state.copy())
            obs_gt_list.append(env.div_ph.copy())
            obs_list.append(obs.copy())
            time_list.append(env.t)

            # Log neurons
            for name, child in network.named_children():
                if name in neuron_dict:
                    neuron_dict[name]["trace"].append(
                        child.trace.detach().clone().view(-1).numpy()
                    )
                    neuron_dict[name]["volt"].append(
                        child.v_cell.detach().clone().view(-1).numpy()
                    ) if hasattr(child, "v_cell") else None
                    neuron_dict[name]["spike"].append(
                        child.spikes.detach().clone().view(-1).numpy()
                    ) if hasattr(child, "spikes") else None
                    neuron_dict[name]["thresh"].append(
                        child.thresh.detach().clone().view(-1).numpy()
                    ) if hasattr(child, "thresh") else None

            # Step the environment
            obs = torch.from_numpy(obs)
            action = network.forward(obs.view(1, 1, -1))
            action = action.numpy()
            obs, _, done, _ = env.step(action)

        # Plot
        plt.plot(time_list, -np.array(state_list)[:, 2], label="Thrust")
        plt.plot(time_list, np.array(state_list)[:, 0], label="Height")
        plt.plot(time_list, np.array(state_list)[:, 1], label="Velocity")
        plt.plot(time_list, np.array(obs_gt_list)[:, 0], label="GT divergence")
        # plt.plot(time_list, np.array(obs_gt_list)[:, 1], label="GT div dot")
        plt.plot(time_list, np.array(obs_list)[:, 0], label="Divergence")
        # plt.plot(time_list, np.array(obs_list)[:, 1], label="Div dot")
        plt.xlabel("Time")
        plt.title(f"Performance starting from {h} m")
        plt.ylim(-5, env.MAX_H + 1)
        plt.legend()
        plt.grid()
        plt.tight_layout()

        # Plot neurons
        fig, ax = plt.subplots(4, 3, figsize=(10, 5))
        for i, (name, recordings) in enumerate(neuron_dict.items()):
            for var, vals in recordings.items():
                if len(vals):
                    for j in range(np.array(vals).shape[1]):
                        ax[j, i].plot(time_list, np.array(vals)[:, j], label=var)
                        ax[j, i].grid(True)
        fig.tight_layout()

        plt.show()
