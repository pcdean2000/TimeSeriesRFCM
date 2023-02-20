import os
import sys
import warnings
warnings.filterwarnings('ignore')

import numpy as np

from glob import glob
from pathlib import Path
from tslearn.utils import from_pyts_dataset
from tslearn.clustering import TimeSeriesKMeans

INTERVAL = int(sys.argv[1])
# INTERVAL = 30

for direction in ["src", "dst"]:
    for dirname in glob(Path(f'timeseries/interval_{INTERVAL}_{direction}*').__str__()):
        print("Dir: ", dirname)
        if os.path.exists(os.path.join(dirname, "tslearn_kmeans.npy")):
            continue
        pyts_dataset = np.load(os.path.join(dirname, "pyts_dataset.npy"))
        pyts_dataset = np.nan_to_num(pyts_dataset)
        print("\tPyts dataset shape: ", pyts_dataset.shape)
        X = from_pyts_dataset(pyts_dataset)
        print("\tTslearn dataset shape: ", X.shape)
        model = TimeSeriesKMeans(n_clusters=10, metric="softdtw", verbose=True, random_state=0, n_jobs=3, max_iter=10)
        y_pred = model.fit_predict(X)
        np.save(os.path.join(dirname, "tslearn_kmeans.npy"), y_pred)
        print("\tTslearn kmeans shape: ", y_pred.shape)
        del pyts_dataset, model, X, y_pred