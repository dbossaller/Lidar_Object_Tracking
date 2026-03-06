import json
import math
import os
from pathlib import Path
import pickle
from random import sample

import numpy as np
import pandas as pd

from scripts.config import PROCESSED_DATA_DIR, RAW_DATA_DIR


def unpack_to_json(
    input_path: Path = RAW_DATA_DIR, output_path: Path = PROCESSED_DATA_DIR, scene_number="004"
):
    frame_dict = unpack_from_pickle(input_path, scene_number)

    processed_dataset = pre_process(frame_dict)

    data_to_json(dataset=processed_dataset, output_dir=output_path, scene_number=scene_number)


"""
Unpack the data from a .pkl file, then sort the data by the keys so the frames are in order.
"""


def unpack_from_pickle(input_path: Path, scene_no: str):
    frame_dict = {}
    for dir in os.scandir(f"{input_path}/" + f"{scene_no}/lidar/"):
        with open(dir, "rb") as file:
            try:
                frame_dict[int(dir.name[0:2])] = pd.DataFrame(pickle.load(file))
            except pickle.UnpicklingError:
                pass

    return dict(sorted(frame_dict.items()))


"""
The raw data has its x and y axes rotated by 45 degrees; this function (and the sub functions below) 
1) Rotate the dataset to align the axes with the 'view' from the front
2) Crops the dataset to a 30x15x5 meter cube
3) Samples the coordinates to make the clusters simpler to calculate.
"""


def pre_process(frame_dict):
    for i in frame_dict.keys():
        frame_dict[i]["xr"], frame_dict[i]["yr"] = (
            rotate_45(frame_dict[i]["x"], frame_dict[i]["y"])[0],
            rotate_45(frame_dict[i]["x"], frame_dict[i]["y"])[1],
        )

    coord_by_frame = {}
    for i in frame_dict.keys():
        crop_dataset(frame_dict[i])

    for i in frame_dict.keys():
        coord_by_frame[i] = sample_rotated_point_cloud(frame_dict[i]).to_numpy()

    return coord_by_frame


def crop_dataset(frame, x_span=15, ymin=0, ymax=15, zmin=0, zmax=5):
    filt = (
        (frame["xr"].apply(abs) > x_span)
        | (frame["yr"] < ymin)
        | (frame["yr"] > ymax)
        | (frame["z"] > zmax)
        | (frame["z"] < zmin)
    )

    frame.drop(frame[filt].index, inplace=True)


def rotate_45(x_val, y_val):
    angle = -math.pi / 4
    x_rot = x_val * math.cos(angle) + y_val * math.sin(angle)
    y_rot = -x_val * math.sin(angle) + y_val * math.cos(angle)
    return x_rot, y_rot


def sample_rotated_point_cloud(frame, num_points=1500):
    try:
        sample_idxs = sample(list(frame.index), num_points)
    except ValueError:
        return frame[["xr", "yr", "z"]]
    sample_idxs = sorted(sample_idxs)
    return frame[["xr", "yr", "z"]].loc[sample_idxs]


"""
Function to take the dataset dictionary and save it as a .json function
"""


def data_to_json(dataset, output_dir, scene_number):
    dataset_as_dict = {}
    for key in dataset.keys():
        dataset_as_dict[int(key)] = dataset[key].tolist()

    with open(f"{output_dir}" + f"/scene{scene_number}.json", "w+") as f:
        json.dump(dataset_as_dict, f, indent=2)


def load_json_data(data_directory, scene_num):
    json_data = read_data_json(data_directory, scene_num)
    return samples_to_dataframes(json_data)


def read_data_json(data_directory, scene_no):
    filename = f"{data_directory}/scene" + f"{scene_no}.json"
    json_data = None
    with open(filename, "r") as dataset:
        json_data = json.load(dataset)
    return json_data


def samples_to_dataframes(raw_json):
    df_dataset = {}
    for key in raw_json.keys():
        df_dataset[int(key)] = raw_json[key]
    return df_dataset


def json_to_numpy(directory, scene_num):
    data_dict = load_json_data(directory, scene_num)
    total_points = len(data_dict) * len(data_dict[0])

    i = 0
    dataset = np.zeros((total_points, 4))
    for key in range(80):
        numpy_data = np.array(data_dict[key])
        for datum in numpy_data:
            frame_datum = np.insert(datum, 0, key * 0.1)
            dataset[i] = frame_datum
            i += 1
    return dataset


if __name__ == "__main__":
    unpack_to_json()
