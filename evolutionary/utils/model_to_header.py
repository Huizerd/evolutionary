import torch

from pysnn.neuron import AdaptiveLIFNeuron

from evolutionary.utils.constructors import build_network


def model_to_header(config, in_file, verbose=2):
    # Build network
    network = build_network(config)
    # Load network parameters
    network.load_state_dict(torch.load(in_file))

    if verbose:
        if network.neuron1 is not None:
            # Write in->hid connection header file
            # Get relevant data
            weights = network.fc1.weight.view(-1).tolist()
            post = network.fc1.weight.size(0)
            pre = network.fc1.weight.size(1)
            # Create string
            string = [
                "//Auto-generated",
                '#include "Connection.h"',
                f"float const w_inhid[] = {{{', '.join([str(w) for w in weights])}}};",
                f"ConnectionConf const conf_inhid = {{{post}, {pre}, w_inhid}};",
            ]
            # Write to file
            with open(f"{config['log location']}connection_conf_inhid.h", "w") as f:
                for line in string:
                    f.write(f"{line}\n")

        if network.neuron1 is not None:
            # Write hid neuron header file
            # Get relevant data
            neuron_type = 1 if isinstance(network.neuron1, AdaptiveLIFNeuron) else 0
            a_v = network.neuron1.alpha_v.view(-1).tolist()
            a_th = (
                network.neuron1.alpha_thresh.view(-1).tolist()
                if isinstance(network.neuron1, AdaptiveLIFNeuron)
                else torch.zeros_like(network.neuron1.alpha_v).view(-1).tolist()
            )
            a_t = network.neuron1.alpha_t.view(-1).tolist()
            d_v = network.neuron1.tau_v.view(-1).tolist()
            d_th = (
                network.neuron1.tau_thresh.view(-1).tolist()
                if isinstance(network.neuron1, AdaptiveLIFNeuron)
                else torch.zeros_like(network.neuron1.tau_v).view(-1).tolist()
            )
            d_t = network.neuron1.tau_t.view(-1).tolist()
            v_rest = network.neuron1.v_rest.item()
            th_rest = (
                (
                    torch.ones_like(network.neuron1.thresh)
                    * network.neuron1.thresh_center
                )
                .view(-1)
                .tolist()
                if isinstance(network.neuron1, AdaptiveLIFNeuron)
                else network.neuron1.thresh.view(-1).tolist()
            )
            size = network.neuron1.spikes.size(-1)
            # Create string
            string = [
                "//Auto-generated",
                '#include "Neuron.h"',
                f"float const a_v_hid[] = {{{', '.join([str(a) for a in a_v])}}};",
                f"float const a_th_hid[] = {{{', '.join([str(a) for a in a_th])}}};",
                f"float const a_t_hid[] = {{{', '.join([str(a) for a in a_t])}}};",
                f"float const d_v_hid[] = {{{', '.join([str(d) for d in d_v])}}};",
                f"float const d_th_hid[] = {{{', '.join([str(d) for d in d_th])}}};",
                f"float const d_t_hid[] = {{{', '.join([str(d) for d in d_t])}}};",
                f"float const th_rest_hid[] = {{{', '.join([str(t) for t in th_rest])}}};",
                f"NeuronConf const conf_hid = {{{neuron_type}, {size}, a_v_hid, a_th_hid, a_t_hid, d_v_hid, d_th_hid, d_t_hid, {v_rest}, th_rest_hid}};",
            ]
            # Write to file
            with open(f"{config['log location']}neuron_conf_hid.h", "w") as f:
                for line in string:
                    f.write(f"{line}\n")

        if network.neuron1 is not None:
            # Write hid->out connection header file
            # Get relevant data
            weights = network.fc2.weight.view(-1).tolist()
            post = network.fc2.weight.size(0)
            pre = network.fc2.weight.size(1)
            # Create string
            string = [
                "//Auto-generated",
                '#include "Connection.h"',
                f"float const w_hidout[] = {{{', '.join([str(w) for w in weights])}}};",
                f"ConnectionConf const conf_hidout = {{{post}, {pre}, w_hidout}};",
            ]
            # Write to file
            with open(f"{config['log location']}connection_conf_hidout.h", "w") as f:
                for line in string:
                    f.write(f"{line}\n")
        else:
            # Write in->out connection header file
            # Get relevant data
            weights = network.fc2.weight.view(-1).tolist()
            post = network.fc2.weight.size(0)
            pre = network.fc2.weight.size(1)
            # Create string
            string = [
                "//Auto-generated",
                '#include "Connection.h"',
                f"float const w_inout[] = {{{', '.join([str(w) for w in weights])}}};",
                f"ConnectionConf const conf_inout = {{{post}, {pre}, w_inout}};",
            ]
            # Write to file
            with open(f"{config['log location']}connection_conf_inout.h", "w") as f:
                for line in string:
                    f.write(f"{line}\n")

        # Write out neuron header file
        # Get relevant data
        neuron_type = 1 if isinstance(network.neuron2, AdaptiveLIFNeuron) else 0
        a_v = network.neuron2.alpha_v.view(-1).tolist()
        a_th = (
            network.neuron2.alpha_thresh.view(-1).tolist()
            if isinstance(network.neuron2, AdaptiveLIFNeuron)
            else torch.zeros_like(network.neuron2.alpha_v).view(-1).tolist()
        )
        a_t = network.neuron2.alpha_t.view(-1).tolist()
        d_v = network.neuron2.tau_v.view(-1).tolist()
        d_th = (
            network.neuron2.tau_thresh.view(-1).tolist()
            if isinstance(network.neuron2, AdaptiveLIFNeuron)
            else torch.zeros_like(network.neuron2.tau_v).view(-1).tolist()
        )
        d_t = network.neuron2.tau_t.view(-1).tolist()
        v_rest = network.neuron2.v_rest.item()
        th_rest = (
            (torch.ones_like(network.neuron2.thresh) * network.neuron2.thresh_center)
            .view(-1)
            .tolist()
            if isinstance(network.neuron2, AdaptiveLIFNeuron)
            else network.neuron2.thresh.view(-1).tolist()
        )
        size = network.neuron2.spikes.size(-1)
        # Create string
        string = [
            "//Auto-generated",
            '#include "Neuron.h"',
            f"float const a_v_out[] = {{{', '.join([str(a) for a in a_v])}}};",
            f"float const a_th_out[] = {{{', '.join([str(a) for a in a_th])}}};",
            f"float const a_t_out[] = {{{', '.join([str(a) for a in a_t])}}};",
            f"float const d_v_out[] = {{{', '.join([str(d) for d in d_v])}}};",
            f"float const d_th_out[] = {{{', '.join([str(d) for d in d_th])}}};",
            f"float const d_t_out[] = {{{', '.join([str(d) for d in d_t])}}};",
            f"float const th_rest_out[] = {{{', '.join([str(t) for t in th_rest])}}};",
            f"NeuronConf const conf_out = {{{neuron_type}, {size}, a_v_out, a_th_out, a_t_out, d_v_out, d_th_out, d_t_out, {v_rest}, th_rest_out}};",
        ]
        # Write to file
        with open(f"{config['log location']}neuron_conf_out.h", "w") as f:
            for line in string:
                f.write(f"{line}\n")

        # Write network header file
        # Get data
        centers = network.in_centers.view(-1).tolist()
        if "place" in network.encoding:
            encoding_type = 1
        elif "offset" in network.encoding:
            encoding_type = 2
        else:
            encoding_type = 0
        if "single" in network.decoding:
            decoding_type = 0
        elif "weighted" in network.decoding:
            decoding_type = 1
        else:
            raise ValueError(f"Incompatible decoding {network.decoding} specified")
        decoding_scale = network.out_scale
        actions = torch.linspace(*network.out_bounds, 20).tolist()
        in_size = 2
        in_enc_size = network.neuron0.spikes.size(-1)
        hid_size = network.neuron1.spikes.size(-1) if network.neuron1 is not None else 0
        out_size = network.neuron2.spikes.size(-1)
        # Create string
        if network.neuron1 is not None:
            string = [
                "//Auto-generated",
                '#include "Network.h"',
                '#include "connection_conf_inhid.h"',
                '#include "connection_conf_hidout.h"',
                '#include "neuron_conf_hid.h"',
                '#include "neuron_conf_out.h"',
                f"float const centers[] = {{{', '.join([str(c) for c in centers])}}};",
                f"float const actions[] = {{{', '.join([str(a) for a in actions])}}};",
                f"NetworkConf const conf = {{{encoding_type}, {decoding_type}, {decoding_scale}, actions, centers, {in_size}, {in_enc_size}, {hid_size}, {out_size}, &conf_inhid, &conf_hid, &conf_hidout, &conf_out}};",
            ]
        else:
            string = [
                "//Auto-generated",
                '#include "Network2.h"',
                '#include "connection_conf_inout.h"',
                '#include "neuron_conf_out.h"',
                f"float const centers[] = {{{', '.join([str(c) for c in centers])}}};",
                f"float const actions[] = {{{', '.join([str(a) for a in actions])}}};",
                f"NetworkConf const conf = {{{encoding_type}, {decoding_type}, {decoding_scale}, actions, centers, {in_size}, {in_enc_size}, {out_size}, &conf_inout, &conf_out}};",
            ]

        # Write to file
        with open(f"{config['log location']}network_conf.h", "w") as f:
            for line in string:
                f.write(f"{line}\n")
