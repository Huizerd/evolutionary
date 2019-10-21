import argparse
import datetime, time
import multiprocessing
import os
import random
from functools import partial
from itertools import chain
from shutil import copyfile

import torch
import yaml
import numpy as np
import pandas as pd
from deap import base, creator, tools

from evolutionary.environment.environment import QuadEnv
from evolutionary.evaluate.evaluate import evaluate
from evolutionary.operators.crossover import crossover_none
from evolutionary.operators.mutation import mutate_call_network
from evolutionary.utils.constructors import build_network_partial
from evolutionary.visualize.vis_network import vis_network, vis_distributions
from evolutionary.visualize.vis_performance import vis_performance
from evolutionary.visualize.vis_population import vis_population, vis_relevant


# Suppress scientific notation
np.set_printoptions(suppress=True)


def main(config, debug=False, no_plot=False):
    # Don't bother with determinism since tournament is stochastic!

    # MP
    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count() // 4)

    # Build network
    network = build_network_partial(config)

    # And init environment
    env = QuadEnv(
        delay=np.random.randint(*config["env"]["delay"]),
        noise=np.random.uniform(*config["env"]["noise"]),
        noise_p=np.random.uniform(*config["env"]["noise p"]),
        thrust_bounds=config["env"]["thrust bounds"],
        thrust_tc=np.random.uniform(*config["env"]["thrust tc"]),
        settle=config["env"]["settle"],
        wind=config["env"]["wind"],
        h0=config["env"]["h0"][0],
        dt=config["env"]["dt"],
        max_t=config["env"]["max time"],
        seed=np.random.randint(config["env"]["seeds"]),
    )

    # Objectives
    # All possible objectives: air time, time to land, final height, final offset,
    # final offset from 5 m, final velocity, unsigned divergence, signed divergence
    valid_objectives = [
        "air time",
        "time to land",
        "final height",
        "final offset",
        "final offset 5m",
        "final velocity",
        "unsigned divergence",
        "signed divergence",
        "dummy",
    ]
    assert len(config["evo"]["objectives"]) == 3, "Only 3 objectives are supported"
    assert all(
        [obj in valid_objectives for obj in config["evo"]["objectives"]]
    ), "Invalid objective"

    # Set up DEAP
    creator.create("Fitness", base.Fitness, weights=config["evo"]["obj weights"])
    creator.create("Individual", list, fitness=creator.Fitness)

    toolbox = base.Toolbox()
    toolbox.register(
        "individual", tools.initRepeat, container=creator.Individual, func=network, n=1
    )
    toolbox.register(
        "population", tools.initRepeat, container=list, func=toolbox.individual
    )
    toolbox.register(
        "evaluate",
        partial(evaluate, valid_objectives, config, env, config["env"]["h0"]),
    )
    toolbox.register("mate", crossover_none)
    toolbox.register(
        "mutate",
        partial(
            mutate_call_network,
            config["evo"]["genes"],
            mutation_rate=config["evo"]["mutation rate"],
        ),
    )
    toolbox.register("select", tools.selNSGA2)
    toolbox.register("map", pool.map)

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean, axis=0)
    stats.register("std", np.std, axis=0)
    stats.register("min", np.min, axis=0)
    stats.register("max", np.max, axis=0)
    stats.register("median", np.median, axis=0)

    logbook = tools.Logbook()
    logbook.header = ("gen", "evals", "avg", "median", "std", "min", "max")

    # Initialize population
    # Pareto front: set of individuals that are not strictly dominated
    # (i.e., better scores for all objectives) by others
    population = toolbox.population(n=config["evo"]["pop size"])
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

    # Update hall of fame
    hof.update(population)

    if not debug:
        # Plot population fitness and its relevant part
        last_pop = vis_population(
            population, hof, config["evo"]["objectives"], no_plot=no_plot
        )
        last_rel = vis_relevant(
            population, hof, config["evo"]["objectives"], no_plot=no_plot
        )

        # Create folders for parameters
        os.makedirs(f"{config['log location']}parameters_0/")
        os.makedirs(f"{config['log location']}hof/")

        # And log the initial performance
        last_pop[0].savefig(f"{config['log location']}population_0.png")
        if last_rel is not None:
            last_rel[0].savefig(f"{config['log location']}relevant_0.png")
        for i, ind in enumerate(population):
            torch.save(
                ind[0].state_dict(),
                f"{config['log location']}parameters_0/individual_{i}.net",
            )

    # Begin the evolution!
    for gen in range(1, config["evo"]["gens"]):
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
            # TODO: can get error because selection is very small (1), below number of extra needed for others
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
        # Population again since we re-evaluated it
        hof.update(population + offspring)

        # Select the population for the next generation
        # from the last generation and its offspring
        population = toolbox.select(population + offspring, config["evo"]["pop size"])

        # Log stuff
        record = stats.compile(population)
        logbook.record(
            gen=gen,
            evals=len(offspring) + len(population),
            **{k: v.round(2) for k, v in record.items()},
        )
        print(logbook.stream)

        if not debug:
            # Plot population fitness and the relevant part of it
            last_pop = vis_population(
                population,
                hof,
                config["evo"]["objectives"],
                last=last_pop,
                no_plot=no_plot,
            )
            last_rel = vis_relevant(
                population,
                hof,
                config["evo"]["objectives"],
                last=last_rel,
                no_plot=no_plot,
            )

            # Log every so many generations
            if not gen % config["log interval"] or gen == config["evo"]["gens"] - 1:
                # Create directory
                if not os.path.exists(f"{config['log location']}parameters_{gen}/"):
                    os.makedirs(f"{config['log location']}parameters_{gen}/")

                # Save population figure
                last_pop[0].savefig(f"{config['log location']}population_{gen}.png")
                if last_rel is not None:
                    last_rel[0].savefig(f"{config['log location']}relevant_{gen}.png")

                # Save parameters of entire population and hall of fame
                for i, ind in enumerate(population):
                    torch.save(
                        ind[0].state_dict(),
                        f"{config['log location']}parameters_{gen}/individual_{i}.net",
                    )
                for i, ind in enumerate(hof):
                    torch.save(
                        ind[0].state_dict(), f"{config['log location']}hof/hof_{i}.net"
                    )

                # Save logbook
                pd.DataFrame(logbook).to_csv(
                    f"{config['log location']}logbook.txt", sep="\t", index=False
                )

    pool.close()


if __name__ == "__main__":
    # Parse input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", choices=["evolve", "test", "distribute"], default="evolve"
    )
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--noplot", action="store_true")
    parser.add_argument("--config", type=str, required=True, default=None)
    parser.add_argument("--tags", nargs="+", default=None)
    parser.add_argument("--parameters", type=str, default=None)
    args = vars(parser.parse_args())

    # Read config file
    with open(args["config"], "r") as cf:
        config = yaml.full_load(cf)

    # Create folders based on time stamp
    start_time = time.time()
    timestamp = datetime.datetime.fromtimestamp(start_time).strftime(
        "%Y-%m-%d_%H:%M:%S"
    )
    config["log location"] += timestamp + "/"
    if (
        not os.path.exists(config["log location"])
        and args["mode"] == "evolve"
        and not args["debug"]
    ):
        os.makedirs(config["log location"])
        # Save config file there for reference
        copyfile(args["config"], config["log location"] + "config.yaml")
        # Save tags here
        assert (
            args["tags"] is not None
        ), "Provide tags for identifying a run! Prefix with @"
        with open(config["log location"] + "tags.txt", "w") as f:
            f.write(" ".join(args["tags"]))

    if args["mode"] == "evolve":
        main(config, args["debug"], args["noplot"])
    elif args["mode"] == "test":
        assert args["parameters"] is not None, "Provide network parameters for testing!"
        config["log location"] = "/".join(args["config"].split("/")[:-1]) + "/"
        config["individual id"] = [
            s.replace(".net", "") for s in args["parameters"].split("/")[-2:]
        ]
        vis_network(config, args["parameters"], args["debug"], args["noplot"])
        vis_performance(config, args["parameters"], args["debug"], args["noplot"])
    elif args["mode"] == "distribute":
        assert args["parameters"] is not None, "Provide network parameters for testing!"
        config["log location"] = "/".join(args["config"].split("/")[:-1]) + "/"
        config["individual id"] = [
            s.replace(".net", "") for s in args["parameters"].split("/")[-2:]
        ]
        vis_distributions(config, args["parameters"], args["debug"], args["noplot"])

    print(f"Duration: {(time.time() - start_time) / 3600:.2f} hours")
