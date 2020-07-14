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

from evolutionary.evaluate.evaluate import evaluate
from evolutionary.operators.crossover import crossover_none
from evolutionary.operators.mutation import mutate_call_network
from evolutionary.utils.constructors import build_network_partial, build_environment
from evolutionary.utils.utils import randomize_env, update
from evolutionary.visualize.vis_population import vis_population


# Suppress scientific notation
np.set_printoptions(suppress=True)


def main(config, verbose):
    # Don't bother with determinism since tournament is stochastic!

    # Set last time to start time
    last_time = start_time

    # MP
    processes = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(processes=processes)

    # Build network
    network = build_network_partial(config)

    # Build environments and randomize
    envs = [build_environment(config) for _ in config["env"]["h0"]]
    for env in envs:
        randomize_env(env, config)

    # Set up DEAP
    creator.create("Fitness", base.Fitness, weights=[-1.0])
    creator.create("Individual", list, fitness=creator.Fitness)

    toolbox = base.Toolbox()
    toolbox.register(
        "individual", tools.initRepeat, container=creator.Individual, func=network, n=1
    )
    toolbox.register(
        "population", tools.initRepeat, container=list, func=toolbox.individual
    )
    toolbox.register(
        "evaluate", partial(evaluate, config, envs, config["env"]["h0"]),
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
    stats.register("avg", np.mean)
    stats.register("median", np.median)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

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
    print(
        f"gen: 0, time past: {time_past:.2f} min, minutes: {minutes:.2f} min, best: {logbook.select('min')[-1]}"
    )

    if verbose:
        # Plot relevant part of population fitness
        last_fig = []
        last_fig.append(vis_population(population, hof, verbose=verbose))

        # Create folders for parameters of individuals
        # Only save hall of fame
        os.makedirs(f"{config['log location']}hof_000/")

        # And log the initial performance
        # Figures
        for i, last in enumerate(last_fig):
            if last[2]:
                last[0].savefig(f"{config['fig location']}population{i}_000.png")
        # Parameters
        for i, ind in enumerate(hof):
            torch.save(
                ind[0].state_dict(),
                f"{config['log location']}hof_000/individual_{i:03}.net",
            )
        # Fitnesses
        pd.DataFrame(
            [ind.fitness.values for ind in hof],
            columns=[f"SSE D{config['evo']['D setpoint']}"],
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
        print(
            f"gen: {gen}, time past: {time_past:.2f} min, minutes: {minutes:.2f} min, best: {logbook.select('min')[-1]}"
        )

        if verbose:
            # Plot relevant part of population fitness
            for i, last, in zip(range(len(last_fig)), last_fig):
                last_fig[i] = vis_population(
                    population, hof, last=last, verbose=verbose
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
                            f"{config['fig location']}population{i}_{gen:03}.png"
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
                    columns=[f"SSE D{config['evo']['D setpoint']}"],
                ).to_csv(
                    f"{config['log location']}hof_{gen:03}/fitnesses.csv",
                    index=False,
                    sep=",",
                )

                # Save logbook
                pd.DataFrame(logbook).to_csv(
                    f"{config['log location']}logbook.csv", index=False, sep=","
                )

    # Close multiprocessing pool
    pool.close()


if __name__ == "__main__":
    # Parse input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", type=int, choices=[0, 1, 2], default=1)
    parser.add_argument("--config", type=str, default="configs/defaults.yaml")
    parser.add_argument("--tags", nargs="+", default=None)
    args = vars(parser.parse_args())

    # Read config file
    # Merge defaults and specifics
    with open("configs/defaults.yaml", "r") as f:
        config = yaml.full_load(f)
    if args["config"] != "configs/defaults.yaml":
        with open(args["config"], "r") as f:
            specifics = yaml.full_load(f)
        config = update(config, specifics)

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
        with open(config["log location"] + "config.yaml", "w") as f:
            yaml.dump(config, f)
        with open(config["log location"] + "tags.txt", "w") as f:
            f.write(" ".join(args["tags"]))

    # Run main
    main(config, args["verbose"])

    print(f"Duration: {(time.time() - start_time) / 3600:.2f} hours")
