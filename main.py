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
from deap.benchmarks.tools import convergence, hypervolume

from evolutionary.evaluate.evaluate import evaluate
from evolutionary.operators.crossover import crossover_none
from evolutionary.operators.mutation import mutate_call_network
from evolutionary.utils.constructors import build_network_partial, build_environment
from evolutionary.utils.model_to_text import model_to_text
from evolutionary.visualize.vis_comparison import vis_comparison
from evolutionary.visualize.vis_network import vis_network
from evolutionary.visualize.vis_performance import vis_performance, vis_disturbance
from evolutionary.visualize.vis_steadystate import vis_steadystate
from evolutionary.visualize.vis_sensitivity import (
    vis_sensitivity,
    vis_sensitivity_complete,
)
from evolutionary.visualize.vis_out_dynamics import vis_out_dynamics
from evolutionary.visualize.vis_statistics import vis_statistics
from evolutionary.visualize.vis_population import vis_population, vis_relevant


# Suppress scientific notation
np.set_printoptions(suppress=True)


def main(config, verbose):
    # Don't bother with determinism since tournament is stochastic!

    # MP
    # Detect GCP or local
    # TODO: why is cloud so much slower than own laptop, even with 4x as many cores?
    if multiprocessing.cpu_count() > 8:
        processes = multiprocessing.cpu_count() - 4
        cloud = True
    else:
        processes = multiprocessing.cpu_count() - 2
        cloud = False
    pool = multiprocessing.Pool(processes=processes)

    # Build network
    network = build_network_partial(config)

    # Build environment and randomize it
    env = build_environment(config)

    # Objectives
    # All possible objectives: air time, time to land, final height, final offset,
    # final offset from 5 m, final velocity, final velocity squared,
    # unsigned divergence, signed divergence, spikes per second (to minimize energy)
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
        "spikes",
        "dummy",
    ]
    assert (
        len(config["evo"]["objectives"]) >= 3
    ), "Only 3 or more objectives are supported"
    assert len(config["evo"]["objectives"]) == len(
        config["evo"]["obj weights"]
    ), "There should be as many weights as objectives"
    assert all(
        [obj in valid_objectives for obj in config["evo"]["objectives"]]
    ), "Invalid objective"

    # Optimal front and reference point for hypervolume
    optimal_front = config["evo"]["obj optimal"]
    hyperref = config["evo"]["obj worst"]
    optim_performance = []

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
    toolbox.register("select", tools.selNSGA2)
    toolbox.register("map", pool.map)

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean, axis=0)
    stats.register("median", np.median, axis=0)
    stats.register("std", np.std, axis=0)
    stats.register("min", np.min, axis=0)
    stats.register("max", np.max, axis=0)

    logbook = tools.Logbook()
    logbook.header = ("gen", "evals", "avg", "median", "std", "min", "max")

    # Initialize population
    # Pareto front: set of individuals that are not strictly dominated
    # (i.e., better scores for all objectives) by others
    population = toolbox.population(n=config["evo"]["pop size"])
    hof = tools.ParetoFront()  # hall of fame!

    # Evaluate initial population
    fitnesses = toolbox.map(toolbox.evaluate, population)
    for ind, fit in zip(population, fitnesses):
        ind.fitness.values = fit

    # This is just to assign the crowding distance (needed for selTournamentDCD())
    # to the individuals, no actual selection is done
    population = toolbox.select(population, len(population))

    # Log first record
    record = stats.compile(population)
    logbook.record(
        gen=0, evals=len(population), **{k: v.round(2) for k, v in record.items()}
    )

    # Update hall of fame
    hof.update(population)

    if verbose:
        # Plot population fitness and its relevant part
        last_pop = []
        last_rel = []
        # Only plot 3D figures when not running on cloud, some problem with matplotlib
        if not cloud:
            for dims in config["evo"]["plot 3D"]:
                last_pop.append(
                    vis_population(
                        population,
                        hof,
                        config["evo"]["objectives"],
                        dims,
                        verbose=verbose,
                    )
                )
        for dims in config["evo"]["plot 2D"]:
            last_rel.append(
                vis_relevant(
                    population, hof, config["evo"]["objectives"], dims, verbose=verbose
                )
            )

        # Create folders for parameters of individuals
        # Only save hall of fame
        os.makedirs(f"{config['log location']}hof_000/")

        # And log the initial performance
        # Figures
        if not cloud:
            for i, last in enumerate(last_pop):
                last[0].savefig(f"{config['fig location']}population{i}_000.png")
        for i, last in enumerate(last_rel):
            if last is not None:
                last[0].savefig(f"{config['fig location']}relevant{i}_000.png")
        # Parameters
        for i, ind in enumerate(hof):
            torch.save(
                ind[0].state_dict(),
                f"{config['log location']}hof_000/individual_{i:03}.net",
            )
        # Fitnesses
        pd.DataFrame(
            [ind.fitness.values for ind in hof], columns=config["evo"]["objectives"]
        ).to_csv(
            f"{config['log location']}hof_000/fitnesses.txt", index=False, sep="\t"
        )

    # Begin the evolution!
    for gen in range(1, config["evo"]["gens"]):
        # Selection: Pareto front + best of the rest
        pareto_fronts = tools.sortNondominated(population, len(population))
        selection = pareto_fronts[0]
        others = list(chain(*pareto_fronts[1:]))
        # We need a multiple of 4 for selTournamentDCD()
        if len(others) % 4:
            others.extend(random.sample(selection, 4 - (len(others) % 4)))
        selection.extend(tools.selTournamentDCD(others, len(others)))

        # Get offspring: mutate selection
        # TODO: maybe add crossover? Which is usually done binary,
        #  so maybe not that useful..
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
        # Also include population, because we re-evaluated it
        hof.update(population + offspring)

        # Select the population for the next generation
        # from the last generation and its offspring
        population = toolbox.select(population + offspring, config["evo"]["pop size"])

        # Log stuff, but don't print!
        record = stats.compile(population)
        logbook.record(
            gen=gen,
            evals=len(offspring) + len(population),
            **{k: v.round(2) for k, v in record.items()},
        )

        # Log convergence (of first front) and hypervolume
        conv = convergence(pareto_fronts[0], optimal_front)
        hyper = hypervolume(pareto_fronts[0], hyperref)
        optim_performance.append([conv, hyper])
        print(f"gen: {gen - 1}, convergence: {conv:.3f}, hypervolume: {hyper:.3f}")

        if verbose:
            # Plot population fitness and the relevant part of it
            # Again, don't print 3D figures when not on laptop
            if not cloud:
                for i, last, dims in zip(
                    range(len(last_pop)), last_pop, config["evo"]["plot 3D"]
                ):
                    last_pop[i] = vis_population(
                        population,
                        hof,
                        config["evo"]["objectives"],
                        dims,
                        last=last,
                        verbose=verbose,
                    )
            for i, last, dims in zip(
                range(len(last_rel)), last_rel, config["evo"]["plot 2D"]
            ):
                last_rel[i] = vis_relevant(
                    population,
                    hof,
                    config["evo"]["objectives"],
                    dims,
                    last=last,
                    verbose=verbose,
                )

            # Log every so many generations
            if not gen % config["log interval"] or gen == config["evo"]["gens"] - 1:
                # Create directory
                if not os.path.exists(f"{config['log location']}hof_{gen:03}/"):
                    os.makedirs(f"{config['log location']}hof_{gen:03}/")

                # Save population figure
                if not cloud:
                    for i, last in enumerate(last_pop):
                        last[0].savefig(
                            f"{config['fig location']}population{i}_{gen:03}.png"
                        )
                for i, last in enumerate(last_rel):
                    if last is not None:
                        last[0].savefig(
                            f"{config['fig location']}relevant{i}_{gen:03}.png"
                        )

                # Save parameters of hall of fame individuals
                for i, ind in enumerate(hof):
                    torch.save(
                        ind[0].state_dict(),
                        f"{config['log location']}hof_{gen:03}/individual_{i:03}.net",
                    )

                # Save fitnesses
                pd.DataFrame(
                    [ind.fitness.values for ind in hof],
                    columns=config["evo"]["objectives"],
                ).to_csv(
                    f"{config['log location']}hof_{gen:03}/fitnesses.txt",
                    index=False,
                    sep="\t",
                )

                # Save logbook
                pd.DataFrame(logbook).to_csv(
                    f"{config['log location']}logbook.txt", index=False, sep="\t"
                )

                # Save optimization performance
                pd.DataFrame(
                    optim_performance, columns=["convergence", "hypervolume"]
                ).to_csv(
                    f"{config['log location']}optim_performance.txt",
                    index=False,
                    sep="\t",
                )

    # Close multiprocessing pool
    pool.close()


if __name__ == "__main__":
    # Parse input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["train", "test", "compare", "analyze", "save"],
        default="train",
    )
    parser.add_argument(
        "--verbose", type=int, choices=[0, 1, 2, 3], default=2
    )  # TODO: 3 for saving values only, not yet implemented
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--tags", nargs="+", default=None)
    parser.add_argument("--parameters", type=str, default=None)
    parser.add_argument("--comparison", type=str, default=None)
    args = vars(parser.parse_args())

    # Modes of execution
    # Training
    if args["mode"] == "train":
        # Read config file
        assert args["config"] is not None, "Training needs a configuration file"
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

    # Testing
    elif args["mode"] == "test":
        # Read config file
        assert args["config"] is not None, "Testing needs a configuration file"
        with open(args["config"], "r") as cf:
            config = yaml.full_load(cf)

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

        # Visualize network parameters
        vis_network(config, args["parameters"], args["verbose"])
        # Visualize landings and network activity
        vis_performance(config, args["parameters"], args["verbose"])
        # Visualize response to a severe disturbance (to see how fast response is)
        vis_disturbance(config, args["parameters"], args["verbose"])
        # Visualize steady-state output for certain inputs
        vis_steadystate(config, args["parameters"], args["verbose"])

    # Comparison
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

    # Analysis
    elif args["mode"] == "analyze":
        # Read config file
        assert args["config"] is not None, "Analysis needs a configuration file"
        with open(args["config"], "r") as cf:
            config = yaml.full_load(cf)

        # Check if folder of parameters was supplied
        assert os.path.isdir(
            args["parameters"]
        ), "Provide a folder of parameters for analysis!"

        # Don't create/save in case of debugging
        if args["verbose"]:
            # Set log location to the one supplied
            folder_id = args["parameters"].split("/")[-2]
            config["log location"] = (
                "/".join(args["config"].split("/")[:-1])
                + "/analysis+"
                + folder_id
                + "+"
            )
            suffix = 0
            while os.path.exists(config["log location"] + str(suffix) + "/"):
                suffix += 1
            config["log location"] += str(suffix) + "/"
            os.makedirs(config["log location"])

        # Visualize output dynamics
        vis_out_dynamics(config, args["parameters"], args["verbose"])
        # Perform sensitivity analysis
        # vis_sensitivity(config, args["parameters"], args["verbose"])
        vis_sensitivity_complete(config, args["parameters"], args["verbose"])
        # Perform statistical analysis
        # vis_statistics(config, args["parameters"], args["verbose"])

    # Save model to text
    elif args["mode"] == "save":
        # Read config file
        assert args["config"] is not None, "Saving needs a configuration file"
        with open(args["config"], "r") as cf:
            config = yaml.full_load(cf)

        # Don't create/save in case of debugging
        if args["verbose"]:
            # Set log location to the one supplied
            individual_id = "_".join(
                [s.replace(".net", "") for s in args["parameters"].split("/")[-2:]]
            )
            config["log location"] = (
                "/".join(args["config"].split("/")[:-1])
                + "/saved+"
                + individual_id
                + "+"
            )
            suffix = 0
            while os.path.exists(config["log location"] + str(suffix) + "/"):
                suffix += 1
            config["log location"] += str(suffix) + "/"
            os.makedirs(config["log location"])

        # Export model to text for use IRL
        model_to_text(config, args["parameters"], args["verbose"])
