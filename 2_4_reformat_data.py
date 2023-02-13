import warnings
warnings.filterwarnings('ignore')

import os
import pickle

import numpy as np
import pandas as pd

from glob import glob
from pathlib import Path
from itertools import combinations

INTERVAL = 1

features = ["packets", "bytes", "flows", "bytes/packets", "flows/(bytes/packets)", "nDstIP", "nSrcPort", "nDstPort"]

original_prefix = f'interval_{INTERVAL}_src_feature'
extended_prefix = ["_reconstructed_STL_trend", "_reconstructed_STL_seasonal", "_reconstructed_STL_combined", "_reconstructed_STL_detrend"]

for prefix in [original_prefix] + [original_prefix + e for e in extended_prefix]:
    print("\nPrefix: ", prefix)
    dirname = os.path.join("timeseries", prefix)
    targetFilename = os.path.join(dirname, "pyts_dataset.npy")
    sampleFilename = os.path.splitext(targetFilename)[0] + "_sample.txt"
    featureFilename = os.path.splitext(targetFilename)[0] + "_feature.txt"
    if os.path.exists(featureFilename):
        continue

    timeseries = {}
    for feature in features:
        print("\tFeature: ", feature)
        result = {}
        for filename in glob(Path(f'{prefix}/*').__str__()):
            print("\t\tFile: ", filename, " " * 5, end="\r")
            with open(filename) as f:
                lines = f.readlines()
                header, data = lines[0], lines[1:]
                index = header.strip().split(',').index(feature)
                value = []
                for line in data:
                    elem = line.strip().split(',')[index]
                    if elem == '':
                        value.append(np.nan)
                    else:
                        value.append(float(elem))
            column = os.path.splitext(os.path.basename(filename))[0]
            result[column] = value
        print()
            
        timeseries[feature] = result

    os.makedirs(os.path.dirname(targetFilename), exist_ok=True)
    pyts_dataset = np.array(np.array(pd.DataFrame.from_dict(timeseries)).tolist())
    np.save(targetFilename, pyts_dataset)

    with open(os.path.join(dirname, 'timeseries_dictionary.pickle'), 'wb') as f:
        pickle.dump(timeseries, f)

    with open(sampleFilename, "w") as f:
        f.write('\n'.join(list(timeseries[features[0]].keys())))
        
    with open(featureFilename, "w") as f:
        f.write('\n'.join(features))
        
    del timeseries, pyts_dataset
        
INTERVAL = 1

features = ["packets", "bytes", "flows", "bytes/packets", "flows/(bytes/packets)", "nDstIP", "nSrcPort", "nDstPort"]

original_prefix = f'interval_{INTERVAL}_src_feature'
extended_prefix = ["_reconstructed_STL_trend", "_reconstructed_STL_seasonal", "_reconstructed_STL_combined", "_reconstructed_STL_detrend"]

for prefix in [original_prefix] + [original_prefix + e for e in extended_prefix]:
    dirname = os.path.join("timeseries", prefix)
    filename = os.path.join(dirname, 'timeseries_dictionary.pickle')
    print("\nFilename: ", filename)
    with open(filename, 'rb') as f:
        timeseries = pickle.load(f)

    targetDirname = dirname.replace("timeseries", "timeseries_feature")

    for feature in combinations(features, 2):
        print("\tFeature: ", feature)
        dirname = os.path.join(targetDirname, f"{feature[0].replace('/', '_')}-{feature[1].replace('/', '_')}")
        targetFilename = os.path.join(dirname, "pyts_dataset.npy")
        featureFilename = os.path.splitext(targetFilename)[0] + "_feature.txt"
        if os.path.exists(featureFilename):
            continue
        
        partial_timeseries = {key: value for key, value in timeseries.items() if key in feature}
        
        os.makedirs(os.path.dirname(targetFilename), exist_ok=True)
        pyts_dataset = np.array(np.array(pd.DataFrame.from_dict(partial_timeseries)).tolist())
        np.save(targetFilename, pyts_dataset)
        
        with open(featureFilename, "w") as f:
            f.write('\n'.join(feature))
        
        del partial_timeseries, pyts_dataset
    
    del timeseries
            
INTERVAL = 1

features = ["packets", "bytes", "flows", "bytes/packets", "flows/(bytes/packets)", "nSrcIP", "nSrcPort", "nDstPort"]

original_prefix = f'interval_{INTERVAL}_dst_feature'
extended_prefix = ["_reconstructed_STL_trend", "_reconstructed_STL_seasonal", "_reconstructed_STL_combined", "_reconstructed_STL_detrend"]

for prefix in [original_prefix] + [original_prefix + e for e in extended_prefix]:
    print("\nPrefix: ", prefix)
    dirname = os.path.join("timeseries", prefix)
    targetFilename = os.path.join(dirname, "pyts_dataset.npy")
    sampleFilename = os.path.splitext(targetFilename)[0] + "_sample.txt"
    featureFilename = os.path.splitext(targetFilename)[0] + "_feature.txt"
    if os.path.exists(featureFilename):
        continue

    timeseries = {}
    for feature in features:
        print("\tFeature: ", feature)
        result = {}
        for filename in glob(Path(f'{prefix}/*').__str__()):
            print("\t\tFile: ", filename, " " * 5, end="\r")
            with open(filename) as f:
                lines = f.readlines()
                header, data = lines[0], lines[1:]
                index = header.strip().split(',').index(feature)
                value = []
                for line in data:
                    elem = line.strip().split(',')[index]
                    if elem == '':
                        value.append(np.nan)
                    else:
                        value.append(float(elem))
            column = os.path.splitext(os.path.basename(filename))[0]
            result[column] = value
        print()
            
        timeseries[feature] = result

    os.makedirs(os.path.dirname(targetFilename), exist_ok=True)
    pyts_dataset = np.array(np.array(pd.DataFrame.from_dict(timeseries)).tolist())
    np.save(targetFilename, pyts_dataset)

    with open(os.path.join(dirname, 'timeseries_dictionary.pickle'), 'wb') as f:
        pickle.dump(timeseries, f)

    with open(sampleFilename, "w") as f:
        f.write('\n'.join(list(timeseries[features[0]].keys())))
        
    with open(featureFilename, "w") as f:
        f.write('\n'.join(features))
    
    del timeseries, pyts_dataset

INTERVAL = 1

features = ["packets", "bytes", "flows", "bytes/packets", "flows/(bytes/packets)", "nSrcIP", "nSrcPort", "nDstPort"]

original_prefix = f'interval_{INTERVAL}_dst_feature'
extended_prefix = ["_reconstructed_STL_trend", "_reconstructed_STL_seasonal", "_reconstructed_STL_combined", "_reconstructed_STL_detrend"]

for prefix in [original_prefix] + [original_prefix + e for e in extended_prefix]:
    dirname = os.path.join("timeseries", prefix)
    filename = os.path.join(dirname, 'timeseries_dictionary.pickle')
    print("\nFilename: ", filename)
    with open(filename, 'rb') as f:
        timeseries = pickle.load(f)

    targetDirname = dirname.replace("timeseries", "timeseries_feature")

    for feature in combinations(features, 2):
        print("\tfeature: ", feature)
        dirname = os.path.join(targetDirname, f"{feature[0].replace('/', '_')}-{feature[1].replace('/', '_')}")
        targetFilename = os.path.join(dirname, "pyts_dataset.npy")
        featureFilename = os.path.splitext(targetFilename)[0] + "_feature.txt"
        if os.path.exists(featureFilename):
            continue
        
        partial_timeseries = {key: value for key, value in timeseries.items() if key in feature}
        
        os.makedirs(os.path.dirname(targetFilename), exist_ok=True)
        pyts_dataset = np.array(np.array(pd.DataFrame.from_dict(partial_timeseries)).tolist())
        np.save(targetFilename, pyts_dataset)

        with open(featureFilename, "w") as f:
            f.write('\n'.join(feature))
        
        del partial_timeseries, pyts_dataset
    
    del timeseries