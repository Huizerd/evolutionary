import argparse
import glob
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def plot_landings(folder):

    folder = Path(folder)
    runs = sorted(Path(folder).rglob("run*.csv"))

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.02)

    for i, run in enumerate(runs):
        data = pd.read_csv(run)
        fig.add_trace(
            go.Scatter(
                name=i,
                x=data["time"],
                y=-data["pos_z"],
                mode="lines",
                line={"color": px.colors.qualitative.Alphabet[i], "width": 2},
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                name=i,
                x=data["time"],
                y=-data["vel_z"],
                mode="lines",
                line={"color": px.colors.qualitative.Alphabet[i], "width": 2},
            ),
            row=2,
            col=1,
        )
        # fig.add_trace(go.Scatter(x=data["time"], y=data["thrust"], mode="lines", line={"color": px.colors.qualitative.Plotly[0], "width": 1}), row=3, col=1)
        # fig.add_trace(go.Scatter(x=data["time"], y=data["div"], mode="lines", line={"color": px.colors.qualitative.Plotly[0], "width": 1}), row=4, col=1)

    fig.update_yaxes(
        title_standoff=30, title_text=r"$h \text{ [m]}$", showgrid=True, row=1, col=1
    )
    fig.update_yaxes(
        title_standoff=20,
        title_text=r"$V \text{ [ms}^{-1}\text{]}$",
        showgrid=True,
        row=2,
        col=1,
    )
    fig.update_yaxes(
        title_standoff=20,
        title_text=r"$T_{sp} \text{ [g]}$",
        showgrid=True,
        row=3,
        col=1,
    )
    fig.update_yaxes(
        title_standoff=30,
        title_text=r"$\hat{D} \text{ [s}^{-1}\text{]}$",
        showgrid=True,
        row=4,
        col=1,
    )
    fig.update_xaxes(range=[0, 7.2], showgrid=True, row=1, col=1)
    fig.update_xaxes(range=[0, 7.2], showgrid=True, row=2, col=1)
    fig.update_xaxes(range=[0, 7.2], showgrid=True, row=3, col=1)
    fig.update_xaxes(
        title_standoff=20,
        range=[0, 7.2],
        title_text=r"$t \text{ [s]}$",
        showgrid=True,
        row=4,
        col=1,
    )
    fig.update_layout(
        title_text="Real-world landings", showlegend=True, template="plotly"
    )
    fig.show()


if __name__ == "__main__":
    # Parse input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", type=str, required=True)
    args = vars(parser.parse_args())

    # Call
    plot_landings(args["folder"])
