def mutate_call_network(genes, types, individual, mutation_rate=1.0):
    # Implemented in networks
    individual[0].mutate(genes, types, mutation_rate)
    return individual
