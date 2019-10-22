import torch

from evolutionary.utils.constructors import build_network


def jit_compile(config, parameters):
    # Load network
    network = build_network(config)
    network.load_state_dict(torch.load(parameters))

    # Script network
    scripted = torch.jit.script(network)
    print(scripted.code)

    # Save scripted
    scripted.save(
        f"{config['log location']}compiled+{'_'.join(config['individual id'])}"
    )
