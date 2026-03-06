import numpy as np
import plotly.graph_objects as go


def main():
    pass


def plot_with_slider(dataset):
    aug_dataset = {}
    for key in dataset.keys():
        aug_dataset[key] = np.concatenate((dataset[key], [[-15, 0, 0], [15, 15, 5]]))

    fig = go.Figure()

    max_frame = max(aug_dataset.keys())
    min_frame = min(aug_dataset.keys())

    for i in range(min_frame, max_frame + 1):
        fig.add_trace(
            go.Scatter3d(
                visible=False,
                mode="markers",
                marker=dict(size=2),
                x=aug_dataset[i][:, 0],
                y=aug_dataset[i][:, 1],
                z=aug_dataset[i][:, 2],
            )
        )
        fig.update_layout(height=500, width=750)
        fig.update_scenes(
            xaxis_range=[-15, 15],
            yaxis_range=[0, 15],
            zaxis_range=[0, 5],
            aspectmode="data",
            overwrite=True,
        )

    fig.data[0].visible = True

    steps = []
    for i in range(len(list(fig.data))):
        step = dict(
            method="update",
            args=[{"visible": [False] * len(list(fig.data))}],
            label=f"{i / 10}s",
        )
        step["args"][0]["visible"][i] = True  # type: ignore
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
    main()
