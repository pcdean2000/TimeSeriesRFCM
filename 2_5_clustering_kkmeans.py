import os
import warnings
warnings.filterwarnings('ignore')

import numpy as np

from glob import glob
from pathlib import Path
from tslearn.utils import from_pyts_dataset
from tslearn.clustering import KernelKMeans

INTERVAL = 30

def dropna(nparray):
    if isinstance(nparray[0], np.ndarray):
        return np.array([dropna(x) for x in nparray])
    else:
        return nparray[~np.isnan(nparray)]
    
for direction in ["src", "dst"]:
    for dirname in glob(Path(f'timeseries/interval_{INTERVAL}_{direction}*').__str__()):
        print("Dir: ", dirname)
        if os.path.exists(os.path.join(dirname, "tslearn_kkmeans.npy")):
            continue
        pyts_dataset = np.load(os.path.join(dirname, "pyts_dataset.npy"))
        pyts_dataset = dropna(pyts_dataset)
        print("\tPyts dataset shape: ", pyts_dataset.shape)
        X = from_pyts_dataset(pyts_dataset)
        print("\tTslearn dataset shape: ", X.shape)
        model = KernelKMeans(n_clusters=10, verbose=True, random_state=10, n_jobs=2, max_iter=10, tol=1e-3, kernel_params={"sigma": 1})
        y_pred = model.fit_predict(X)
        np.save(os.path.join(dirname, "tslearn_kkmeans.npy"), y_pred)
        print("\tTslearn kmeans shape: ", y_pred.shape)
        del pyts_dataset, model, X, y_pred