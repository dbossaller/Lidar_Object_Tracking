import json
import math
import os
from pathlib import Path
import pickle
from random import sample

import pandas as pd
import typer

from lidar_object_tracking.config import PROCESSED_DATA_DIR, RAW_DATA_DIR

app = typer.Typer()


@app.command()
def main(
    input_path: Path = RAW_DATA_DIR,
    output_path: Path = PROCESSED_DATA_DIR,
):
    frame_dict = unpack_dataset(input_path, "004")

    for i in frame_dict.keys():
        frame_dict[i]["xr"], frame_dict[i]["yr"] = (
            rotate_45(frame_dict[i]["x"], frame_dict[i]["y"])[0],
            rotate_45(frame_dict[i]["x"], frame_dict[i]["y"])[1],
        )

    for i in frame_dict.keys():
        frame_dict[i].drop(
            frame_dict[i][
                (frame_dict[i]["xr"].apply(abs) > 25)
                | (frame_dict[i]["yr"] < 0)
                | (frame_dict[i]["yr"] > 25)
                | (frame_dict[i]["z"] > 5)
                | (frame_dict[i]["z"] < 0)
            ].index,
            inplace=True,
        )

    sub_sample = [sample(list(frame_dict[i].index), 1500) for i in range(len(frame_dict))]

    coord_by_frame = {
        i: frame_dict[i][["xr", "yr", "z"]].loc[sub_sample[i]] for i in range(len(frame_dict))
    }

    data_to_json(coord_by_frame, output_path, "004")


def unpack_dataset(input_path, scene_no):
    frame_dict = {}
    for dir in os.scandir(f"{input_path}/" + f"{scene_no}/lidar/"):
        with open(dir, "rb") as file:
            try:
                frame_dict[int(dir.name[0:2])] = pd.DataFrame(pickle.load(file))
            except:
                pass

    return dict(sorted(frame_dict.items()))


def rotate_45(x_val, y_val):
    angle = -math.pi / 4
    x_rot = x_val * math.cos(angle) + y_val * math.sin(angle)
    y_rot = -x_val * math.sin(angle) + y_val * math.cos(angle)
    return x_rot, y_rot


def data_to_dict(dataset):
    scene_dict = {}
    for i in dataset.keys():
        scene_dict[i] = dataset[i].to_dict(orient="index")
        return scene_dict


def data_to_json(dataset, output_dir, scene_number):
    dataset = data_to_dict(dataset=dataset)
    with open(f"{output_dir}" + f"/scene{scene_number}.json", "w+") as f:
        json.dump(dataset, f, indent=2)


if __name__ == "__main__":
    app()
