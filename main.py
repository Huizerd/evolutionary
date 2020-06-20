#! usr/bin/env python

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
from evolutionary.utils.model_to_header import model_to_header
from evolutionary.utils.utils import randomize_env
from evolutionary.visualize.vis_network import vis_network
from evolutionary.visualize.vis_performance import vis_performance, vis_disturbance
from evolutionary.visualize.vis_steadystate import vis_steadystate
from evolutionary.visualize.vis_sensitivity import (
    vis_sensitivity_complete,
    vis_sensitivity_complete_4m,
)
from evolutionary.visualize.vis_population import vis_relevant


# Suppress scientific notation
np.set_printoptions(suppress=True)


def main(config, verbose):
    # Don't bother with determinism since tournament is stochastic!

    # Set last time to start time
    last_time = start_time

    # MP
    processes = multiprocessing.cpu_count() // 2
    pool = multiprocessing.Pool(processes=processes)

    # Build network
    network = build_network_partial(config)

    # Build environments and randomize
    envs = [build_environment(config) for _ in config["env"]["h0"]]
    for env in envs:
        randomize_env(env, config)

    # Objectives
    # Time to land, final height, final velocity, spikes per second
    valid_objectives = [
        "time to land",
        "time to land scaled",
        "final height",
        "final velocity",
        "final velocity squared",
        "spikes",
        "SSE D0.5",
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
        partial(evaluate, valid_objectives, config, envs, config["env"]["h0"]),
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

    # Update hall of fame
    hof.update(population)

    # Log first record
    record = stats.compile(population)
    logbook.record(
        gen=0, evals=len(population), **{k: v.round(2) for k, v in record.items()}
    )

    # Log convergence (of first front) and hypervolume
    pareto_fronts = tools.sortNondominated(population, len(population))
    current_time = time.time()
    minutes = (current_time - last_time) / 60
    last_time = time.time()
    time_past = (current_time - start_time) / 60
    conv = convergence(pareto_fronts[0], optimal_front)
    hyper = hypervolume(pareto_fronts[0], hyperref)
    optim_performance.append([0, time_past, minutes, conv, hyper])
    print(
        f"gen: 0, time past: {time_past:.2f} min, minutes: {minutes:.2f} min, convergence: {conv:.3f}, hypervolume: {hyper:.3f}"
    )

    if verbose:
        # Plot relevant part of population fitness
        last_fig = []
        for dims in config["evo"]["plot"]:
            last_fig.append(
                vis_relevant(
                    population, hof, config["evo"]["objectives"], dims, verbose=verbose
                )
            )

        # Create folders for parameters of individuals
        # Only save hall of fame
        os.makedirs(f"{config['log location']}hof_000/")

        # And log the initial performance
        # Figures
        for i, last in enumerate(last_fig):
            if last[2]:
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
        ).to_csv(f"{config['log location']}hof_000/fitnesses.csv", index=False, sep=",")

    # Begin the evolution!
    for gen in range(1, config["evo"]["gens"]):
        # Randomize environments (in-place) for this generation
        # Each individual in a generation experiences the same environments,
        # but re-seeding per individual is not done to prevent identically-performing
        # agents (and thus thousands of HOFs, due to stepping nature of SNNs)
        for env in envs:
            randomize_env(env, config)

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
        pareto_fronts = tools.sortNondominated(population, len(population))
        current_time = time.time()
        minutes = (current_time - last_time) / 60
        last_time = time.time()
        time_past = (current_time - start_time) / 60
        conv = convergence(pareto_fronts[0], optimal_front)
        hyper = hypervolume(pareto_fronts[0], hyperref)
        optim_performance.append([gen, time_past, minutes, conv, hyper])
        print(
            f"gen: {gen}, time past: {time_past:.2f} min, minutes: {minutes:.2f} min, convergence: {conv:.3f}, hypervolume: {hyper:.3f}"
        )

        if verbose:
            # Plot relevant part of population fitness
            for i, last, dims in zip(
                range(len(last_fig)), last_fig, config["evo"]["plot"]
            ):
                last_fig[i] = vis_relevant(
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
                for i, last in enumerate(last_fig):
                    if last[2]:
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
                    f"{config['log location']}hof_{gen:03}/fitnesses.csv",
                    index=False,
                    sep=",",
                )

                # Save logbook
                pd.DataFrame(logbook).to_csv(
                    f"{config['log location']}logbook.csv", index=False, sep=","
                )

                # Save optimization performance
                pd.DataFrame(
                    optim_performance,
                    columns=[
                        "gen",
                        "time past",
                        "minutes",
                        "convergence",
                        "hypervolume",
                    ],
                ).to_csv(
                    f"{config['log location']}optim_performance.csv",
                    index=False,
                    sep=",",
                )

    # Close multiprocessing pool
    pool.close()


if __name__ == "__main__":
    # Parse input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["train", "test", "analyze", "analyze4m", "save"],
        default="train",
    )
    parser.add_argument("--verbose", type=int, choices=[0, 1, 2], default=1)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--tags", nargs="+", default=None)
    parser.add_argument("--parameters", type=str, default=None)
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

        # Perform sensitivity analysis
        vis_sensitivity_complete(config, args["parameters"], args["verbose"])

    # Analysis from 4m
    elif args["mode"] == "analyze4m":
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
                + "/analysis4m+"
                + folder_id
                + "+"
            )
            suffix = 0
            while os.path.exists(config["log location"] + str(suffix) + "/"):
                suffix += 1
            config["log location"] += str(suffix) + "/"
            os.makedirs(config["log location"])

        # Perform sensitivity analysis
        vis_sensitivity_complete_4m(config, args["parameters"], args["verbose"])

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
        model_to_header(config, args["parameters"], args["verbose"])
