# Lidar Object Tracking

This seeks to use basic machine learning techniques to identify and track moving objects in LiDAR data. First the data is aggregated into a numpy array where the first coordinate is time (in seconds). From there, a DBSCAN clustering algorithm is used to track clusters (i.e. objects in the field of view) over time. From there, a Kalman filter is used approximate the velocity of the cluster, and only the clusters whose speed exceed a threshold are published.

## Project Organization
```
root
├── data
│   ├── processed                   <-- Processed data in .json 
│   └── raw                         <-- Raw data in .pkl format
├── notebooks                       <-- Jupyter Notebooks w/ examples
│   ├── 01-Unpacking_dataset.ipynb  <-- Notebook unpacking and saving the dataset as a .json file
│   ├── 02-clustering.ipynb         <-- Notebook using the DBSCAN algorithm to cluster the points
│   └── 03-kalman_filter.ipynb      <-- Notebook which combines the clustering and kalman filter to identify clusters that are moving.        
├── scripts
│   ├── config.py                   <-- Contains path variables and constants        
│   ├── dataset.py                  <-- Functions for modifying data; the main function unpacks the raw dataset and saves it as a .json
│   ├── features.py                 <-- Functions that run DBSCAN for clustering and run the Kalman filter
│   └── plots.py                    <-- Function for plotting data with a slider to track objects.
```
## Data Source:
The data used in this project come from the [PandaSet](https://pandaset.org/) dataset made available by Scale AI inc. The data should be downloaded to the '.data/raw/' directory. The dataset contains far more data than simply the LiDAR data we use.
