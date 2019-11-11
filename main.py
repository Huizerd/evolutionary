import argparse
import time
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

from evolutionary.evaluate.evaluate import evaluate
from evolutionary.operators.crossover import crossover_none
from evolutionary.operators.mutation import mutate_call_network
from evolutionary.utils.constructors import build_network_partial, build_environment
from evolutionary.visualize.vis_comparison import vis_comparison
from evolutionary.visualize.vis_network import vis_network
from evolutionary.visualize.vis_performance import vis_performance, vis_disturbance
from evolutionary.visualize.vis_steadystate import vis_steadystate
from evolutionary.visualize.vis_population import vis_population, vis_relevant


# Suppress scientific notation
np.set_printoptions(suppress=True)


def main(config, verbose):
    # Don't bother with determinism since tournament is stochastic!

    # MP
    # Detect GCP or local
    if multiprocessing.cpu_count() > 8:
        processes = multiprocessing.cpu_count() - 4
        cloud = True
    else:
        processes = multiprocessing.cpu_count() - 2
        cloud = False
    pool = multiprocessing.Pool(processes=processes)

    # Build network
    network = build_network_partial(config)

    # And environment
    env = build_environment(config)

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
        "final velocity squared",
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
            config["evo"]["types"],
            mutation_rate=config["evo"]["mutation rate"],
        ),
    )
    toolbox.register("preselect", tools.selTournamentDCD)
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
    fitnesses, envs = toolbox.map(toolbox.evaluate, population)
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

    if verbose:
        # Plot population fitness and its relevant part
        # Doesn't work in the cloud for some reason
        if not cloud:
            last_pop = vis_population(
                population,
                hof,
                config["evo"]["objectives"],
                len(config["env"]["h0"]),
                verbose=verbose,
            )
        last_rel = vis_relevant(
            population,
            hof,
            config["evo"]["objectives"],
            len(config["env"]["h0"]),
            verbose=verbose,
        )

        # Create folders for parameters
        os.makedirs(f"{config['log location']}hof_0/")

        # And log the initial performance
        # Figures
        if not cloud:
            last_pop[0].savefig(f"{config['fig location']}population_0.png")
        if last_rel is not None:
            last_rel[0].savefig(f"{config['fig location']}relevant_0.png")
        # Parameters
        for i, ind in enumerate(hof):
            torch.save(
                ind[0].state_dict(), f"{config['log location']}hof_0/individual_{i}.net"
            )
        # Fitnesses
        pd.DataFrame(
            [ind.fitness.values for ind in hof], columns=config["evo"]["objectives"]
        ).to_csv(f"{config['log location']}hof_0/fitnesses.txt", index=False, sep="\t")

    # Begin the evolution!
    for gen in range(1, config["evo"]["gens"]):
        # Pre-selection through tournament based on dominance and crowding distance
        selection = toolbox.preselect(population, len(population))
        # pareto_fronts = tools.sortNondominated(population, len(population))
        # selection = pareto_fronts[0]
        # others = list(chain(*pareto_fronts[1:]))
        # if len(others) % 4:
        #     others.extend(random.sample(selection, 4 - (len(others) % 4)))
        # selection.extend(tools.selTournamentDCD(others, len(others)))

        # Get offspring: mutate selection
        # TODO: maybe add crossover
        offspring = [toolbox.mutate(toolbox.clone(ind)) for ind in selection]
        # offspring = [toolbox.mutate(toolbox.clone(ind)) for ind in selection[:len(population)]]

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

        if verbose:
            # Plot population fitness and the relevant part of it
            if not cloud:
                last_pop = vis_population(
                    population,
                    hof,
                    config["evo"]["objectives"],
                    len(config["env"]["h0"]),
                    last=last_pop,
                    verbose=verbose,
                )
            last_rel = vis_relevant(
                population,
                hof,
                config["evo"]["objectives"],
                len(config["env"]["h0"]),
                last=last_rel,
                verbose=verbose,
            )

            # Log every so many generations
            if not gen % config["log interval"] or gen == config["evo"]["gens"] - 1:
                # Create directory
                if not os.path.exists(f"{config['log location']}hof_{gen}/"):
                    os.makedirs(f"{config['log location']}hof_{gen}/")

                # Save population figure
                if not cloud:
                    last_pop[0].savefig(f"{config['fig location']}population_{gen}.png")
                if last_rel is not None:
                    last_rel[0].savefig(f"{config['fig location']}relevant_{gen}.png")

                # Save parameters of hall of fame individuals
                for i, ind in enumerate(hof):
                    torch.save(
                        ind[0].state_dict(),
                        f"{config['log location']}hof_{gen}/individual_{i}.net",
                    )

                # Save fitnesses
                pd.DataFrame(
                    [ind.fitness.values for ind in hof],
                    columns=config["evo"]["objectives"],
                ).to_csv(
                    f"{config['log location']}hof_{gen}/fitnesses.txt",
                    index=False,
                    sep="\t",
                )

                # Save logbook
                pd.DataFrame(logbook).to_csv(
                    f"{config['log location']}logbook.txt", index=False, sep="\t"
                )

    pool.close()


if __name__ == "__main__":
    # Parse input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "test", "compare"], default="train")
    parser.add_argument(
        "--verbose", type=int, choices=[0, 1, 2, 3], default=2
    )  # 3 for saving values, not yet implemented
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--tags", nargs="+", default=None)
    parser.add_argument("--parameters", nargs="+", default=None)
    parser.add_argument("--comparison", type=str, default=None)
    args = vars(parser.parse_args())

    # Modes of execution
    if args["mode"] == "train":
        # Read config file
        assert args["config"] is not None, "Training needs a single configuration file"
        with open(args["config"], "r") as cf:
            config = yaml.full_load(cf)

        # Check if we supplied tags for identification
        assert args["tags"] is not None, "Provide tags for identifying a run!"
        # Start time
        start_time = time.time()

        # Don't create/save in case of debugging
        if args["verbose"]:
            # Create folders, add suffix if necessary
            config["log location"] += "+".join(args["tags"]) + "+"
            suffix = 0
            while os.path.exists(config["log location"] + str(suffix) + "/"):
                suffix += 1
            config["log location"] += str(suffix) + "/"
            config["fig location"] = config["log location"] + "population_figs/"
            os.makedirs(config["log location"])
            os.makedirs(config["fig location"])

            # Save config file and tags there
            copyfile(args["config"], config["log location"] + "config.yaml")
            with open(config["log location"] + "tags.txt", "w") as f:
                f.write(" ".join(args["tags"]))

        # Run main
        main(config, args["verbose"])

        print(f"Duration: {(time.time() - start_time) / 3600:.2f} hours")
    elif args["mode"] == "test":
        # Read config file
        assert args["config"] is not None, "Testing needs a single configuration file"
        with open(args["config"], "r") as cf:
            config = yaml.full_load(cf)

        # Check if single set of parameters was supplied
        assert len(args["parameters"]) == 1, "Provide a single network for testing!"
        args["parameters"] = args["parameters"][0]

        # Don't create/save in case of debugging
        if args["verbose"]:
            # Set log location to the one supplied
            individual_id = "_".join(
                [s.replace(".net", "") for s in args["parameters"].split("/")[-2:]]
            )
            config["log location"] = (
                "/".join(args["config"].split("/")[:-1])
                + "/test+"
                + individual_id
                + "+"
            )
            suffix = 0
            while os.path.exists(config["log location"] + str(suffix) + "/"):
                suffix += 1
            config["log location"] += str(suffix) + "/"
            os.makedirs(config["log location"])
        vis_network(config, args["parameters"], args["verbose"])
        vis_performance(config, args["parameters"], args["verbose"])
        vis_disturbance(config, args["parameters"], args["verbose"])
        vis_steadystate(config, args["parameters"], args["verbose"])
    elif args["mode"] == "compare":
        # Load config files
        assert args["comparison"] is not None, "Comparison needs a yaml file"
        with open(args["comparison"], "r") as cf:
            comparison = yaml.full_load(cf)
            configs = []
            for conf in comparison["configs"]:
                with open(conf, "r") as ccf:
                    configs.append(yaml.full_load(ccf))

        # Check if we supplied tags for identification
        assert args["tags"] is not None, "Provide tags for identifying a run!"
        # Start time
        start_time = time.time()

        # Don't create/save in case of debugging
        if args["verbose"]:
            # Create folders, add suffix if necessary
            comparison["log location"] += "+".join(args["tags"]) + "+"
            suffix = 0
            while os.path.exists(comparison["log location"] + str(suffix) + "/"):
                suffix += 1
            comparison["log location"] += str(suffix) + "/"
            os.makedirs(comparison["log location"])

            # Save comparison file and tags there
            copyfile(args["comparison"], comparison["log location"] + "comparison.yaml")
            with open(comparison["log location"] + "tags.txt", "w") as f:
                f.write(" ".join(args["tags"]))

        # Perform comparison
        vis_comparison(configs, comparison, args["verbose"])
