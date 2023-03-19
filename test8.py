import math
import timeit

import numpy as np
import pandas as pd

from itertools import combinations
from multiprocessing import Pool
from random import randint
from scipy.spatial.distance import pdist, squareform

def initial_worker(cluster_sizes, number_entries, number_features):
    global _cluster_sizes, _number_entries, _number_features
    _cluster_sizes = cluster_sizes
    _number_entries = number_entries
    _number_features = number_features
    
def task_similarity_func(args):
    global _cluster_sizes, _number_entries, _number_features
    a, b = args
    # print(a, b)
    return np.sum([math.e ** -(5 * _cluster_sizes[i][a[i]] / _number_entries) for i in np.where(a == b)[0]], dtype=np.float16) / _number_features
    # return np.sum([_cluster_sizes[i][a[i]] for i in np.where(a == b)[0]], dtype=np.float16)

def main():
    n_entries = 10

    df = pd.DataFrame.from_dict(
        {
            "col1": [randint(0, 2) for _ in range(n_entries)],
            "col2": [randint(0, 2) for _ in range(n_entries)],
            "col3": [randint(0, 2) for _ in range(n_entries)],
            "col4": [randint(0, 2) for _ in range(n_entries)],
        }
    )
    df.index = [f"row{i}" for i in range(n_entries)]
    
    cluster_sizes = [df.iloc[:, i].value_counts() for i in range(len(df.columns))]
    number_entries = len(df.index)
    number_features = len(df.columns)
    
    with Pool(processes=4, initializer=initial_worker, initargs=(cluster_sizes, number_entries, number_features)) as pool:
        results = pool.map(task_similarity_func, combinations(df.values, 2))
    
    # results = math.e ** -(5 * np.array(results) / number_entries) / number_features
    print(pd.DataFrame(squareform(results), index=df.index, columns=df.index))
    
    def similarity_func(a, b):
        return np.sum([math.e ** -(5 * cluster_sizes[i][a[i]] / number_entries) for i in np.where(a == b)[0]], dtype=np.float16) / number_features

    print(pd.DataFrame(squareform(pdist(df, similarity_func)) + np.identity(number_entries), index=df.index, columns=df.index))

if __name__ == "__main__":
    print(timeit.timeit(main, number=1))