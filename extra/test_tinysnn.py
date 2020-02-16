import torch

from evolutionary.network.snn import SNN


if __name__ == "__main__":
    # Build config
    config_gen = {"double neurons": True, "double actions": False}
    config_env = {"thrust bounds": [-0.8, 0.5]}
    config_snn = {
        "neuron": ["adaptive", "regular"],
        "thresh": [0.9, 1.0],
        "v rest": [0.0, 0.0],
        "alpha v": [1.0, 1.0],
        "alpha t": [1.0, 1.0],
        "alpha thresh": [1.0, 1.0],
        "tau v": [0.5, 0.5],
        "tau t": [0.5, 0.5],
        "tau thresh": [0.5, 0.5],
        "refrac": [0, 0],
        "dt": 1,
        "delay": 0,
        "input scale": 0.1,
        "input offset": 1.0,
        "output scale": 1.0,
        "output offset": 0.0,
        "decoding": "trace",
    }
    config = {"env": config_env, "snn": config_snn}
    config.update(config_gen)

    # Build network
    snn = SNN(2, 4, 1, config)
    snn.fc1.weight.copy_(torch.tensor([[0.1, 0.1], [0.2, 0.2], [0.5, 0.5], [1.0, 1.0]]))
    snn.fc2.reset_weights(distribution="constant", gain=0.5)

    # Inputs to network
    inputs = torch.tensor([2.0]).view(1, 1, -1)

    # Forward network
    output = snn.forward(inputs)

    # Print everything
    print("Network inputs:")
    print(inputs)
    print()

    print("Encoded network inputs:")
    print(snn.input)
    print()

    print("Input -> hidden weights:")
    print(snn.fc1.weight)
    print()

    print("Hidden layer inputs:")
    print()

    print("Hidden layer voltages:")
    print(snn.neuron1.v_cell)
    print()

    print("Hidden layer thresholds:")
    print(snn.neuron1.thresh)
    print()

    print("Hidden layer spikes:")
    print(snn.neuron1.spikes)
    print()

    print("Hidden layer trace:")
    print(snn.neuron1.trace)
    print()

    print("Hidden -> output weights:")
    print(snn.fc2.weight)
    print()

    print("Output layer inputs:")
    print()

    print("Output layer voltages:")
    print(snn.neuron2.v_cell)
    print()

    print("Output layer thresholds:")
    print(snn.neuron2.thresh)
    print()

    print("Output layer spikes:")
    print(snn.neuron2.spikes)
    print()

    print("Output layer trace:")
    print(snn.neuron2.trace)
    print()

    print("Decoded network output:")
    print(output)
    print()
