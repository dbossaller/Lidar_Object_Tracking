from pathlib import Path

from loguru import logger
import numpy as np
from sklearn.cluster import DBSCAN
from tqdm import tqdm
import typer
from pykalman import KalmanFilter

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
        raise ValueError(f'{label} is not a valid label for the dataset. Your label should be an integer between 0 and {max(labels)}')
    
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
        center_coords[int(key)] = np.array([data_cluster[key][:,0].mean(),
                              data_cluster[key][:,1].mean(),
                              data_cluster[key][:,2].mean()])
    return center_coords

def run_kalman(cluster_data_i):
    frame_rate = 0.1

    # rows: x, y, dx, dy
    state_transition_matrix = np.array([[1, 0, frame_rate , 0],
                             [0, 1, 0, frame_rate],
                             [0, 0, 1, 0],
                             [ 0, 0 , 0 ,1]])
    obs_matrix = np.array([[1,0,0,0],[0,1,0,0]])

    kf = KalmanFilter(transition_matrices = state_transition_matrix,
                    observation_matrices= obs_matrix)
    measurements = cluster_data_i.T

    kf.em(measurements, n_iter=5)
    kf.filter(measurements)
    (smoothed_state_means,_) = kf.smooth(measurements)

    return smoothed_state_means


def identify_moving_clusters(dataset):
    moving_clusters ={}
    moving_estimates = {}
    labels = dbscan_clustering_labels(dataset)
    cluster_points = {}
    cluster_centers = {}
    xy_centers = {}
    for i in list(np.unique(labels)):
        if i >=0:
            cluster_points[i] = pull_clusters_fill_frames(dataset, labels, label = i)
            cluster_centers = find_cluster_centers(cluster_points[i])
            # Reduce to an xy coordinate system since objects on the street generally don't change vertical position.
            x_centers = [cluster_centers[key][0] for key in cluster_centers.keys()]
            y_centers = [cluster_centers[key][1] for key in cluster_centers.keys()]

            xy_centers[i] = np.array([x_centers,y_centers])

    # Run Kalman Filter
    state_estimates = {}
    cluster_list = []
    for i in xy_centers.keys():
        state_estimates[i] = run_kalman(xy_centers[i])
        x_vel_est_i = state_estimates[i][:,2]
        y_vel_est_i = state_estimates[i][:,3]

        speed = [(x_vel_est_i[idx]**2+  y_vel_est_i[idx]**2)**0.5 for idx in range(len(x_vel_est_i))]

        avg_speed = np.mean(speed)
    
        if avg_speed >= 0.5:
            cluster_list.append(i)
            x_pos_est_i = state_estimates[i][:,0]
            y_pos_est_i = state_estimates[i][:,0]
            print(f'Cluster {i} is moving at a rate of {avg_speed} m/s.')
            moving_estimates[i] = [x_pos_est_i, y_pos_est_i, x_vel_est_i, y_vel_est_i]
            moving_clusters[i] = cluster_points[i]
        else:
            print(f'Cluster {i} has an average speed of {avg_speed} m/s. It is likely not a moving cluster')
        
    return moving_clusters, moving_estimates

if __name__ == "__main__":
    app()
