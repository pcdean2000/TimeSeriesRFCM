import warnings
warnings.filterwarnings('ignore')

import os

import numpy as np

from glob import glob
from pathlib import Path
from tslearn.utils import from_pyts_dataset
from tslearn.clustering import TimeSeriesKMeans

INTERVAL = 30

for filename in glob(f'interval{INTERVAL}_reconstructed_STL_trend_ts_feature/200702111400/*/*.npy'):
    print(filename)
    
    labelFilename = filename.replace('reconstructed_STL_trend_ts_feature', 'reconstructed_STL_trend_ts_feature_clustered')
    pickleFilename = labelFilename.replace('.npy', '.pkl')
    if os.path.isfile(pickleFilename):
        print('Already done')
        continue
    
    X = np.load(filename)
    X = from_pyts_dataset(X)
    print(X.shape)
    model = TimeSeriesKMeans(n_clusters=3, metric="softdtw", max_iter=10, random_state=0)
    model.fit(X)
    print(np.asarray(np.unique(model.labels_, return_counts=True)).T)
    
    os.makedirs(os.path.dirname(labelFilename), exist_ok=True)
    os.makedirs(os.path.dirname(pickleFilename), exist_ok=True)
    
    np.save(labelFilename, model.labels_)
    model.to_pickle(pickleFilename)