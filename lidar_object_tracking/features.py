from pathlib import Path

from loguru import logger
import numpy as np
from sklearn.cluster import DBSCAN
from tqdm import tqdm
import typer

from lidar_object_tracking.config import PROCESSED_DATA_DIR

app = typer.Typer()


@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    input_path: Path = PROCESSED_DATA_DIR ,
    output_path: Path = PROCESSED_DATA_DIR ,
    # -----------------------------------------
):
    # ---- REPLACE THIS WITH YOUR OWN CODE ----
    logger.info("Generating features from dataset...")
    for i in tqdm(range(10), total=10):
        if i == 5:
            logger.info("Something happened for iteration 5.")
    logger.success("Features generation complete.")
    # -----------------------------------------

def dbscan_clustering_labels(dataset, epsilon = 1, min_cluster_size=50):
    db = DBSCAN(eps=epsilon, min_samples=min_cluster_size)

    clusters = db.fit(dataset)

    labels = clusters.labels_
    print(f'Found {len(np.unique((labels)))-1} clusters')

    return labels

def pull_data_with_labels(dataset,labels, label):
    if label not in labels:
        raise ValueError(f'{label} is not a valid label for the dataset.')
    
    filt = (labels == label)

    data_with_label = dataset[filt]

    label_dict = {}
    for value in np.unique(data_with_label[:,0]):
        label_dict[int(value*10)] = data_with_label[data_with_label[:,0] == value][:,[1,2,3]]
    return dict(sorted(label_dict.items()))

def fill_in_frames(dataset_dict:dict):
    key_list = sorted(dataset_dict.keys())

    for idx in range(min(key_list), max(key_list)+1):
        try:
            dataset_dict[idx]
        except KeyError:
            dataset_dict[idx] = dataset_dict[idx - 1]

    return dict(sorted(dataset_dict.items()))

def pull_clusters_fill_frames(dataset, labels, label):
        filled_data_dict = fill_in_frames(pull_data_with_labels(dataset, labels, label))
        return filled_data_dict

def find_cluster_centers(data_cluster:dict):
    center_coords = {}
    for key in data_cluster.keys():
        center_coords[key] = np.array([data_cluster[key][:,0].mean(),
                              data_cluster[key][:,1].mean(),
                              data_cluster[key][:,2].mean()])
    return center_coords
if __name__ == "__main__":
    app()
