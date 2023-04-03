import argparse
import json
import math
import os
from glob import glob
from itertools import combinations
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import squareform

def initial_worker(cluster_sizes, number_entries, number_features):
    global _cluster_sizes, _number_entries, _number_features
    _cluster_sizes = cluster_sizes
    _number_entries = number_entries
    _number_features = number_features
    
def task_similarity_func(args):
    global _cluster_sizes, _number_entries, _number_features
    a, b = args
    return np.sum([math.e ** -(5 * _cluster_sizes[i][a[i]] / _number_entries) for i in np.where(a == b)[0]], dtype=np.float16) / _number_features

def main(interval, n_jobs=4):
    for direction in ['src']:
        for workingDir in glob(Path(f"timeseries_feature/interval_{interval}_{direction}*/").__str__()):
            print("Working directory:", workingDir)
            labelfile = os.path.join(workingDir, "rfcm_results.csv")
            typefile = os.path.join(workingDir, "types.json")
            targetfile = os.path.join(workingDir, "similarity_matrix.csv")
            if os.path.exists(targetfile):
                print("\tTarget file exists, skip")
                continue
            
            print("\tLoading data...", " " * 50, end="\r", flush=True)
            
            with open(typefile) as f:
                types = json.load(f)
                
            df = pd.read_csv(labelfile, index_col=0, dtype=types)
            df = df.astype(np.float16)
            
            print("\tCalculating similarity matrix...", " " * 50, end="\r", flush=True)
            
            cluster_sizes = [df.iloc[:, i].value_counts() for i in range(len(df.columns))]
            number_entries = len(df.index)
            number_features = len(df.columns)
            
            with Pool(processes=n_jobs, initializer=initial_worker, initargs=(cluster_sizes, number_entries, number_features)) as pool:
                results = pool.map(task_similarity_func, combinations(df.values, 2))

            print("\tSaving similarity matrix...", " " * 50, flush=True)
            
            similarity_matrix = pd.DataFrame(squareform(results), index=df.index, columns=df.index, dtype=np.float16)
            similarity_matrix.to_csv("timeseries_feature/interval_30_src_feature/similarity_matrix.csv")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="2_7_similarity.py",
        description="Calculate similarity matrix for each interval and direction"
    )
    parser.add_argument('interval', type=int, help="Interval of time series")
    parser.add_argument('-j', '--n_jobs', type=int, default=4, help="Number of jobs")
    args = parser.parse_args()
    
    print("-" * 80)
    print("Doing interval:", args.interval)
    print("With n_jobs:", args.n_jobs)
    print("-" * 80)
    
    main(interval=args.interval, n_jobs=args.n_jobs)