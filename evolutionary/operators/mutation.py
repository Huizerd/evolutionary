def mutate_call_network(genes, individual, mutation_rate=1.0):
    # Implemented in networks
    individual[0].mutate(genes, mutation_rate)
    return individual
