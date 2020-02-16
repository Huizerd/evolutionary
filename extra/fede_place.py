import torch
import matplotlib.pyplot as plt

from evolutionary.utils.utils import sigmoid

if __name__ == "__main__":
    input_bounds = [-10.0, 10.0]
    input_size = 11
    div = torch.linspace(-50.0, 50.0, 1001).view(-1, 1, 1)
    # div.clamp_(*input_bounds)
    centers_linear = torch.linspace(*input_bounds, input_size).view(1, 1, -1)
    centers_sigmoid = sigmoid(
        centers_linear,
        torch.tensor(input_bounds[0]),
        torch.tensor(input_bounds[1] - input_bounds[0]),
        torch.tensor(sum(input_bounds) / 2),
        torch.tensor(0.5),
    )
    centers_cubed = torch.pow(centers_linear, 3) / (input_bounds[1] ** 2)

    print(centers_cubed.view(-1).tolist())

    fig1, ax1 = plt.subplots()
    ax1.plot(centers_linear.view(-1).numpy(), centers_sigmoid.view(-1).numpy())
    ax1.plot(centers_linear.view(-1).numpy(), centers_cubed.view(-1).numpy())
    ax1.grid()

    encoded = 1.0 * torch.exp(
        -((div.clamp_(*input_bounds) - centers_cubed) ** 2) / (4.0 * 1.0)
    )

    fig2, ax2 = plt.subplots()
    for i, center in enumerate(centers_linear.view(-1).tolist()):
        ax2.plot(
            div.view(-1).numpy(),
            encoded[..., i].view(-1).numpy(),
            label=f"Neuron {i}, center: {center}",
        )

    ax2.grid()
    ax2.legend()

    plt.show()
