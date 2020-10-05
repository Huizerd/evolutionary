def mutate_call_network(genes, mutations, limits, decay, individual, mutation_rate=1.0):
    # Implemented in networks
    individual[0].mutate(genes, mutations, limits, decay, mutation_rate)
    return individual
