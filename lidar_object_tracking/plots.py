from pathlib import Path

from loguru import logger
import plotly.graph_objects as go
from tqdm import tqdm
import typer

from lidar_object_tracking.config import FIGURES_DIR, PROCESSED_DATA_DIR

app = typer.Typer()


@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    input_path: Path = PROCESSED_DATA_DIR,
    output_path: Path = FIGURES_DIR,
    # -----------------------------------------
):
    # ---- REPLACE THIS WITH YOUR OWN CODE ----
    logger.info("Generating plot from data...")
    for i in tqdm(range(10), total=10):
        if i == 5:
            logger.info("Something happened for iteration 5.")
    logger.success("Plot generation complete.")
    # -----------------------------------------


def plot_with_slider(dataset):
    fig = go.Figure()

    for i in range(80):

        fig.add_trace(
            go.Scatter3d(
                visible=False,
                mode="markers",
                marker=dict(size=2),
                x=dataset[i]["xr"],
                y=dataset[i]["yr"],
                z=dataset[i]["z"],
            )
        )
        fig.update_layout(height=500, width=750)
        fig.update_scenes(
            xaxis_range=[-10, 10],
            yaxis_range=[0, 10],
            zaxis_range=[0, 5],
            aspectmode="data",
            overwrite=True
        )

    fig.data[0].visible = True

    steps = []
    for i in range(len(list(fig.data))):
        step = dict(
            method="update",
            args=[{"visible": [False] * len(list(fig.data))}],
            label=f"{i / 10}s",
        )
        step["args"][0]["visible"][i] = True
        steps.append(step)

    sliders = [
        dict(
            active=0,
            currentvalue={
                "prefix": "Frame: ",
            },
            pad={"t": 80},
            steps=steps,
        )
    ]

    fig.update_layout(sliders=sliders)

    fig.show()


if __name__ == "__main__":
    app()
