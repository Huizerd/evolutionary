def mutate_call_network(individual, mutation_rate=1.0):
    # Implemented in networks
    individual[0].mutate(mutation_rate)
    return individual
