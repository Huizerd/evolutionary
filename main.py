import argparse
import random
from functools import partial
from itertools import chain

import torch
import numpy as np
from deap import base, creator, tools
from scoop import futures

# TODO: look at package structure, maybe more in (init) files instead of separate files and folders?
# TODO: why does it work with bindsnet? The structure?
from evolutionary.network.ann import ANN
from evolutionary.evaluate.eval_hover import eval_hover
from evolutionary.operators.crossover import crossover_none
from evolutionary.operators.mutation import mutate_call_network
from evolutionary.visualize.vis_hover import vis_hover
from evolutionary.visualize.vis_population import vis_population


# TODO: set seeds here in case we go for full determinism
# np.random.seed(0)
# random.seed(0)

# Suppress scientific notation
np.set_printoptions(suppress=True)

# Set up DEAP
# TODO: need we move this for configurability?
MUTATION_RATE = 0.3
NGEN = 250
MU = 100

creator.create("Fitness", base.Fitness, weights=(-1.0, -1.0, -1.0))  # minimize all
creator.create("Individual", list, fitness=creator.Fitness)

toolbox = base.Toolbox()
# TODO: check the need for n=1 here
# TODO: we need initialization here, why didn't kirk?
toolbox.register(
    "individual",
    tools.initRepeat,
    container=creator.Individual,
    func=partial(ANN, 2, 8, 1),
    n=1,
)
toolbox.register(
    "population", tools.initRepeat, container=list, func=toolbox.individual
)
toolbox.register("evaluate", eval_hover)
toolbox.register("mate", crossover_none)
toolbox.register("mutate", partial(mutate_call_network, mutation_rate=MUTATION_RATE))
toolbox.register("select", tools.selNSGA2)
toolbox.register("map", futures.map)  # for SCOOP

# TODO: or move into main because it needs to be reset for each call of main?
stats = tools.Statistics(lambda ind: ind.fitness.values)
stats.register("avg", np.mean, axis=0)
stats.register("std", np.std, axis=0)
stats.register("min", np.min, axis=0)
stats.register("max", np.max, axis=0)
stats.register("median", np.median, axis=0)

logbook = tools.Logbook()
logbook.header = ("gen", "evals", "avg", "median", "std", "min", "max")


def main(seed=None):
    # Not enough for determinism (since tournament is non-deterministic!)
    # TODO: keep or remove? Or adjust tournament?
    random.seed(seed)
    np.random.seed(seed)

    # Initialize population
    # Pareto front: set of individuals that are not strictly dominated
    # (i.e., better scores for all objectives) by others
    population = toolbox.population(n=MU)
    hof = tools.ParetoFront()  # hall of fame!

    # Evaluate population
    fitnesses = toolbox.map(toolbox.evaluate, population)
    for ind, fit in zip(population, fitnesses):
        ind.fitness.values = fit

    # This is just to assign the crowding distance to the individuals,
    # no actual selection is done
    # "Crowding distance" appears to be something related to NSGA-II (which is select())
    population = toolbox.select(population, len(population))

    # Log first record
    record = stats.compile(population)
    logbook.record(
        gen=0, evals=len(population), **{k: v.round(2) for k, v in record.items()}
    )
    print(logbook.stream)

    # Plot population fitness
    last_fig = vis_population(population)

    # Update hall of fame
    hof.update(population)

    # TODO: Kirk has some folder creation + initial weight saving here

    # Begin the evolution!
    for gen in range(1, NGEN):
        # Get Pareto front
        # sortNondominated() returns a list of "fronts",
        # of which the first is the actual Pareto front
        pareto_fronts = tools.sortNondominated(population, len(population))

        # Select Pareto-optimal individuals
        selection = pareto_fronts[0]

        # Group the others together in a single list
        others = list(chain(*pareto_fronts[1:]))

        # Tournament below needs others to be a multiple of 4,
        # so extend with already selected individuals
        if len(others) % 4:
            others.extend(random.sample(selection, 4 - (len(others) % 4)))

        # Extend the selection based on a tournament played by the others
        # DCD stands for dominance and crowding distance
        # (which is used in case there is no strict dominance)
        # Select k-out-of-k because we want as much offspring as possible,
        # next generation will later be selected from population + offspring
        selection.extend(tools.selTournamentDCD(others, len(others)))

        # Get offspring: mutate selection,
        # possibly cutting off those we added for the tournament
        # This works, since everything is sorted from best to worst
        offspring = [
            toolbox.mutate(toolbox.clone(ind)) for ind in selection[: len(population)]
        ]

        # Re-evaluate last generation/population, because their conditions are random
        # and we want to test each individual against as many as possible
        fitnesses = toolbox.map(toolbox.evaluate, population)
        for ind, fit in zip(population, fitnesses):
            ind.fitness.values = fit

        # And evaluate the entire new offspring, for the same reason
        fitnesses = toolbox.map(toolbox.evaluate, offspring)
        for ind, fit in zip(offspring, fitnesses):
            ind.fitness.values = fit

        # Update the hall of fame with the offspring,
        # so we get the best of population + offspring in there
        hof.update(offspring)

        # Select the population for the next generation
        # from the last generation and its offspring
        population = toolbox.select(population + offspring, MU)

        # TODO: Kirk does another sort here to plot the Pareto front

        # Log stuff
        record = stats.compile(population)
        logbook.record(
            gen=gen,
            evals=len(offspring) + len(population),
            **{k: v.round(2) for k, v in record.items()},
        )
        print(logbook.stream)

        # Plot population fitness
        last_fig = vis_population(population, last=last_fig)

        # TODO: Kirk does some weight saving here

    # TODO: and some final saving/logging here
    # Save hall of fame
    for i, ind in enumerate(hof):
        torch.save(ind[0].state_dict(), f"logs/hof_{i}.net")

    # TODO: Save last fig + indicate hall of fame?
    last_fig.save("logs/final.png")


if __name__ == "__main__":
    # Parse input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["evolve", "test"], default="evolve")
    parser.add_argument("--weights", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    args = vars(parser.parse_args())

    if args["mode"] == "evolve":
        main(args["seed"])
    elif args["mode"] == "test":
        assert args["weights"] is not None, "Provide weights for testing!"
        vis_hover(args["weights"])
