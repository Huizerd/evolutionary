import torch
import pandas as pd

from pysnn.neuron import LIFNeuron, AdaptiveLIFNeuron

from evolutionary.utils.constructors import build_network


def model_to_text(config, in_file, verbose=2):
    # Build network
    network = build_network(config)
    # Load network parameters
    network.load_state_dict(torch.load(in_file))

    if verbose:
        # Write in->hid weight file
        pd.DataFrame(network.fc1.weight.numpy()).to_csv(
            f"{config['log location']}weights_inhid.txt",
            header=False,
            index=False,
            sep=" ",
        )
        # Write hid neuron file
        pd.DataFrame(
            [
                [
                    network.neuron1.alpha_v.item(),
                    network.neuron1.alpha_thresh.item()
                    if isinstance(network.neuron1, AdaptiveLIFNeuron)
                    else 0.0,
                    network.neuron1.alpha_t.item(),
                    network.neuron1.tau_v.item(),
                    network.neuron1.tau_thresh.item()
                    if isinstance(network.neuron1, AdaptiveLIFNeuron)
                    else 0.0,
                    network.neuron1.tau_t.item(),
                    network.neuron1.v_rest.item(),
                    network.neuron1.thresh_center.item(),
                    1 if isinstance(network.neuron1, AdaptiveLIFNeuron) else 0,
                ]
            ]
        ).to_csv(
            f"{config['log location']}neuron_hid.txt",
            header=False,
            index=False,
            sep=" ",
        )
        # Write hid->out weight file
        pd.DataFrame(network.fc2.weight.numpy()).to_csv(
            f"{config['log location']}weights_hidout.txt",
            header=False,
            index=False,
            sep=" ",
        )
        # Write out neuron file
        pd.DataFrame(
            [
                [
                    network.neuron2.alpha_v.item(),
                    network.neuron2.alpha_thresh.item()
                    if isinstance(network.neuron2, AdaptiveLIFNeuron)
                    else 0.0,
                    network.neuron2.alpha_t.item(),
                    network.neuron2.tau_v.item(),
                    network.neuron2.tau_thresh.item()
                    if isinstance(network.neuron2, AdaptiveLIFNeuron)
                    else 0.0,
                    network.neuron2.tau_t.item(),
                    network.neuron2.v_rest.item(),
                    network.neuron2.thresh_center.item(),
                    1 if isinstance(network.neuron2, AdaptiveLIFNeuron) else 0,
                ]
            ]
        ).to_csv(
            f"{config['log location']}neuron_out.txt",
            header=False,
            index=False,
            sep=" ",
        )
        # Write network file
        pd.DataFrame(
            [
                "weights_inhid.txt",
                "neuron_hid.txt",
                "weights_hidout.txt",
                "neuron_out.txt",
            ]
        ).to_csv(
            f"{config['log location']}network.txt", header=False, index=False, sep=" "
        )
