import os
import warnings
warnings.filterwarnings('ignore')

import numpy as np

from glob import glob
from pathlib import Path
from tslearn.utils import from_pyts_dataset
from tslearn.clustering import TimeSeriesKMeans

INTERVAL = 1

for dirname in glob(Path(f'timeseries_feature/interval_{INTERVAL}_src_*/*').__str__()):
    print("Dir: ", dirname)
    if os.path.exists(os.path.join(dirname, "tslearn_kmeans.npy")):
        continue
    pyts_dataset = np.load(os.path.join(dirname, "pyts_dataset.npy"))
    print("\tPyts dataset shape: ", pyts_dataset.shape)
    X = from_pyts_dataset(pyts_dataset)
    print("\tTslearn dataset shape: ", X.shape)
    model = TimeSeriesKMeans(n_clusters=10, metric="softdtw", metric_params={"gamma": .01}, verbose=True, random_state=0, max_iter=20)
    model.fit(X)
    y_pred = model.labels_
    np.save(os.path.join(dirname, "tslearn_kmeans.npy"), y_pred)
    print("\tTslearn kmeans shape: ", y_pred.shape)
    model.to_hdf5(os.path.join(dirname, "tslearn_kmeans.h5"))
    print("\tTslearn kmeans model saved")
    del pyts_dataset, model, X, y_pred