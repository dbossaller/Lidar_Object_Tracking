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

    coord_by_frame = {}
    for i in frame_dict.keys():
        crop_dataset(frame_dict[i])

    for i in frame_dict.keys():
        coord_by_frame[i] = sample_rotated_point_cloud(frame_dict[i])

def unpack_dataset(input_path: Path, scene_no: str):
    frame_dict = {}
    for dir in os.scandir(f"{input_path}/" + f"{scene_no}/lidar/"):
        with open(dir, "rb") as file:
            try:
                frame_dict[int(dir.name[0:2])] = pd.DataFrame(pickle.load(file))
            except:
                pass

    return dict(sorted(frame_dict.items()))

def crop_dataset(frame, xmin = -25, xmax = 25, ymin = 0, ymax = 25, zmin=1, zmax=5):
    filt = (
    (frame["xr"] < xmin)|
    (frame["xr"] > xmax)|
    (frame["yr"] < ymin)|
    (frame["yr"] > ymax)|
    (frame["z"] > zmax)|
    (frame["z"] < zmin))
    
    frame.drop(
        frame[filt
            ].index,
            inplace=True,
        )

def rotate_45(x_val, y_val):
    angle = -math.pi / 4
    x_rot = x_val * math.cos(angle) + y_val * math.sin(angle)
    y_rot = -x_val * math.sin(angle) + y_val * math.cos(angle)
    return x_rot, y_rot

def sample_rotated_point_cloud(frame, num_points = 1500):
    sample_idxs = sample(list(frame.index), num_points)
    return frame[['xr','yr','z']].loc[sample_idxs]


def data_to_dict(dataset):
    scene_dict = {}
    for i in dataset.keys():
        scene_dict[i] = dataset[i].to_dict(orient="index")
        return scene_dict


def data_to_json(dataset, output_dir, scene_number):
    dataset = data_to_dict(dataset)
    with open(f"{output_dir}" + f"/scene{scene_number}.json", "w+") as f:
        json.dump(dataset, f, indent=2)


if __name__ == "__main__":
    app()
