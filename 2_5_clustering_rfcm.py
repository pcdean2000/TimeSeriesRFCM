import os
import sys
import warnings
warnings.filterwarnings('ignore')

import numpy as np

from glob import glob
from pathlib import Path
from rfcm import RFCM

INTERVAL = int(sys.argv[1])
# INTERVAL = 30

for direction in ["src", "dst"]:
    for dirname in glob(Path(f'timeseries_feature/interval_{INTERVAL}_{direction}*/*').__str__()):
        print("Dir: ", dirname)
        if os.path.exists(os.path.join(dirname, "rfcm_label.npy")):
            continue
        pyts_dataset = np.load(os.path.join(dirname, "pyts_dataset.npy"))
        print("\tPyts dataset shape: ", pyts_dataset.shape)
        model = RFCM(n_clusters=10, max_iter=10, random_state=10, n_jobs=4)
        model.fit(pyts_dataset)
        y_pred = model.labels_
        np.save(os.path.join(dirname, "rfcm_label.npy"), y_pred)
        print("\tRFCM label shape: ", y_pred.shape)
        del pyts_dataset, model, y_pred