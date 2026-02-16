import math
import os
from pathlib import Path
import pickle

from loguru import logger
import pandas as pd
from tqdm import tqdm
import typer

from lidar_object_tracking.config import PROCESSED_DATA_DIR, RAW_DATA_DIR

app = typer.Typer()


@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    input_path: Path = RAW_DATA_DIR,
    output_path: Path = PROCESSED_DATA_DIR / "dataset.csv",
    # ----------------------------------------------
):
    # ---- REPLACE THIS WITH YOUR OWN CODE ----
    logger.info("Processing dataset...")
    for i in tqdm(range(10), total=10):
        if i == 5:
            logger.info("Something happened for iteration 5.")
    logger.success("Processing dataset complete.")
    # -----------------------------------------


def unpack_dataset(input_path, scene_no):
    frame_dict = {}
    for dir in os.scandir(input_path + f"{scene_no}/lidar/"):
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


def df_to_json(dataset, output_path):
    return dataset.to_json(output_path, indent=2)


if __name__ == "__main__":
    app()
