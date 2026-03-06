# Lidar Object Tracking

This seeks to use basic machine learning techniques to identify and track moving objects in LiDAR data.

## Project Organization
```
root
├── data
│   ├── processed           <-- Processed data in .json 
│   └── raw                 <-- Raw data in .pkl format
├── notebooks               <-- Jupyter Notebooks w/ examples
│   ├── 01-Unpacking_dataset.ipynb
│   ├── 02-clustering.ipynb     
│   └── 03-kalman_filter.ipynb              
├── scripts
│   ├── config.py           <-- Contains path variables and constants        
│   ├── dataset.py          <-- Functions for modifying data
│   ├── features.py         <-- Clustering scripts
│   └── plots.py            <-- Function for plotting data w/ slider
```
## Data Source:
The data used in this project come from the [PandaSet](https://pandaset.org/) dataset made available by Scale AI inc. The data should be downloaded to the '.data/raw/' directory. The dataset contains far more data than simply the LiDAR data we use.
