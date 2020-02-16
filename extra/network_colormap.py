import argparse

import pandas as pd
import numpy as np


def add_colormap(folder, colormap):
    vertices = pd.read_csv(folder + "network_vertices.csv", sep=",")
    edges = pd.read_csv(folder + "network_edges.csv", sep=",")
    rates = pd.read_csv(folder + "rates.csv", sep=",")

    # Vertices
    rgb = {"R": [], "G": [], "B": []}
    bins = np.linspace(0.0, 30.0, colormap.shape[0] + 1)

    k = 0
    for i in range(vertices.shape[0]):
        if vertices["id"][i] < 0:
            rgb["R"].append(colormap[0, 0] * 255)
            rgb["G"].append(colormap[0, 1] * 255)
            rgb["B"].append(colormap[0, 2] * 255)
        else:
            bin = np.digitize(rates["mean_time"][k], bins) - 1
            rgb["R"].append(colormap[bin, 0] * 255)
            rgb["G"].append(colormap[bin, 1] * 255)
            rgb["B"].append(colormap[bin, 2] * 255)
            k += 1

    vertices_new = pd.concat([vertices, pd.DataFrame(rgb)], axis=1)
    vertices_new.to_csv(folder + "network_vertices_new.csv", index=False, sep=",")

    # Edges
    rgb = {"R": [], "G": [], "B": []}
    opacity = {"opacity": []}

    for i in range(edges.shape[0]):
        if abs(edges["lw_raw"][i]) < 0.05:
            opacity["opacity"].append(0)
        else:
            opacity["opacity"].append(1)
        if edges["lw_raw"][i] < 0:
            rgb["R"].append(colormap[0, 0] * 255)
            rgb["G"].append(colormap[0, 1] * 255)
            rgb["B"].append(colormap[0, 2] * 255)
        else:
            rgb["R"].append(colormap[-1, 0] * 255)
            rgb["G"].append(colormap[-1, 1] * 255)
            rgb["B"].append(colormap[-1, 2] * 255)

    edges_new = pd.concat([edges, pd.DataFrame(rgb), pd.DataFrame(opacity)], axis=1)
    edges_new.to_csv(folder + "network_edges_new.csv", index=False, sep=",")


if __name__ == "__main__":
    # Parse input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", type=str, required=True)
    args = vars(parser.parse_args())

    # Colormap
    # Custom viridis of 9 colors
    viridis = pd.read_csv("tryouts/viridis.csv", sep=",", header=None).to_numpy()
    viridis9 = viridis[1::2, :]

    # Call
    add_colormap(args["folder"], viridis9)
