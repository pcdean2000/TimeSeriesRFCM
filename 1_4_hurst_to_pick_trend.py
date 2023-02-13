import gc
import math
import numba
import threading
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from glob import glob
from numpy import pi
from nolds import hurst_rs as hurst
from multiprocessing.pool import ThreadPool as Pool

# @numba.njit()
class SSA(object):
    
    __supported_types = (pd.Series, np.ndarray, list)
    
    def __init__(self, tseries, L, save_mem=True):
        """
        Decomposes the given time series with a singular-spectrum analysis. Assumes the values of the time series are
        recorded at equal intervals.
        
        Parameters
        ----------
        tseries : The original time series, in the form of a Pandas Series, NumPy array or list. 
        L : The window length. Must be an integer 2 <= L <= N/2, where N is the length of the time series.
        save_mem : Conserve memory by not retaining the elementary matrices. Recommended for long time series with
            thousands of values. Defaults to True.
        
        Note: Even if an NumPy array or list is used for the initial time series, all time series returned will be
        in the form of a Pandas Series or DataFrame object.
        """
        
        # Tedious type-checking for the initial time series
        if not isinstance(tseries, self.__supported_types):
            raise TypeError("Unsupported time series object. Try Pandas Series, NumPy array or list.")
        
        # Checks to save us from ourselves
        self.N = len(tseries)
        if not 2 <= L <= self.N/2:
            raise ValueError("The window length must be in the interval [2, N/2].")
        
        self.L = L
        self.orig_TS = pd.Series(tseries)
        self.K = self.N - self.L + 1
        
        # Embed the time series in a trajectory matrix
        self.X = np.array([self.orig_TS.values[i:L+i] for i in range(0, self.K)]).T
        
        # Decompose the trajectory matrix
        self.U, self.Sigma, VT = np.linalg.svd(self.X)
        self.d = np.linalg.matrix_rank(self.X)
        
        self.TS_comps = np.zeros((self.N, self.d))
        
        if not save_mem:
            # Construct and save all the elementary matrices
            self.X_elem = np.array([ self.Sigma[i]*np.outer(self.U[:,i], VT[i,:]) for i in range(self.d) ])

            # Diagonally average the elementary matrices, store them as columns in array.           
            for i in range(self.d):
                X_rev = self.X_elem[i, ::-1]
                self.TS_comps[:,i] = [X_rev.diagonal(j).mean() for j in range(-X_rev.shape[0]+1, X_rev.shape[1])]
            
            self.V = VT.T
        else:
            # Reconstruct the elementary matrices without storing them
            for i in range(self.d):
                X_elem = self.Sigma[i]*np.outer(self.U[:,i], VT[i,:])
                X_rev = X_elem[::-1]
                self.TS_comps[:,i] = [X_rev.diagonal(j).mean() for j in range(-X_rev.shape[0]+1, X_rev.shape[1])]
            
            self.X_elem = "Re-run with save_mem=False to retain the elementary matrices."
            
            # The V array may also be very large under these circumstances, so we won't keep it.
            self.V = "Re-run with save_mem=False to retain the V matrix."
        
        # Calculate the w-correlation matrix.
        self.calc_wcorr()
            
    def components_to_df(self, n=0):
        """
        Returns all the time series components in a single Pandas DataFrame object.
        """
        if n > 0:
            n = min(n, self.d)
        else:
            n = self.d
        
        # Create list of columns - call them F0, F1, F2, ...
        cols = ["F{}".format(i) for i in range(n)]
        return pd.DataFrame(self.TS_comps[:, :n], columns=cols, index=self.orig_TS.index)
            
    
    def reconstruct(self, indices):
        """
        Reconstructs the time series from its elementary components, using the given indices. Returns a Pandas Series
        object with the reconstructed time series.
        
        Parameters
        ----------
        indices: An integer, list of integers or slice(n,m) object, representing the elementary components to sum.
        """
        if isinstance(indices, int): indices = [indices]
        
        ts_vals = self.TS_comps[:,indices].sum(axis=1)
        return pd.Series(ts_vals, index=self.orig_TS.index)
    
    def calc_wcorr(self):
        """
        Calculates the w-correlation matrix for the time series.
        """
             
        # Calculate the weights
        w = np.array(list(np.arange(self.L)+1) + [self.L]*(self.K-self.L-1) + list(np.arange(self.L)+1)[::-1])
        
        def w_inner(F_i, F_j):
            return w.dot(F_i*F_j)
        
        # Calculated weighted norms, ||F_i||_w, then invert.
        F_wnorms = np.array([w_inner(self.TS_comps[:,i], self.TS_comps[:,i]) for i in range(self.d)])
        F_wnorms = F_wnorms**-0.5
        
        # Calculate Wcorr.
        self.Wcorr = np.identity(self.d)
        for i in range(self.d):
            for j in range(i+1,self.d):
                self.Wcorr[i,j] = abs(w_inner(self.TS_comps[:,i], self.TS_comps[:,j]) * F_wnorms[i] * F_wnorms[j])
                self.Wcorr[j,i] = self.Wcorr[i,j]
    
    def plot_wcorr(self, min=None, max=None):
        """
        Plots the w-correlation matrix for the decomposed time series.
        """
        if min is None:
            min = 0
        if max is None:
            max = self.d
        
        if self.Wcorr is None:
            self.calc_wcorr()
        
        ax = plt.imshow(self.Wcorr)
        plt.xlabel(r"$\tilde{F}_i$")
        plt.ylabel(r"$\tilde{F}_j$")
        plt.colorbar(ax.colorbar, fraction=0.045)
        ax.colorbar.set_label("$W_{i,j}$")
        plt.clim(0,1)
        
        # For plotting purposes:
        if max == self.d:
            max_rnge = self.d-1
        else:
            max_rnge = max
        
        plt.xlim(min-0.5, max_rnge+0.5)
        plt.ylim(max_rnge+0.5, min-0.5)

# @numba.njit()
def get_boundary(N):
    M = math.log2(N)
    log_log_M = math.log(math.log(M))

    # 95% confidence interval
    lower_bound = 0.5 - math.exp(-7.33 * log_log_M + 4.21)
    upper_bound = 0.5 + math.exp(-7.20 * log_log_M + 4.04)

    # 99% confidence interval
    # lower_bound = 0.5 - math.exp(-7.19 * log_log_M + 4.34)
    # upper_bound = 0.5 + math.exp(-7.51 * log_log_M + 4.58)
    
    return lower_bound, upper_bound

# @numba.njit()
def drop_perturbations(df, L, lower_bound, upper_bound, max_components=50):
    for column in df.columns.difference(["timeStart", "SrcIP"]):
        F_ssa = SSA(df[column].values, L)
        components = F_ssa.components_to_df(max_components)
        for i in range(max_components):
            if lower_bound <= hurst(components[f"F{i}"]) <= upper_bound:
                break
        df[column] = F_ssa.reconstruct(list(range(i)))
        
# max_components = 50

def job(filename, max_components = 50):
# for filename in glob(Path('interval1/200702111400/*/*').__str__()):
    try:
        targetFilename = filename.replace("interval1", "reconstructed")
        if os.path.exists(targetFilename):
            # continue
            return
        df = pd.read_csv(filename)
        L = df.shape[0] // 2
        lower_bound, upper_bound = get_boundary(df.shape[0])
        # drop_perturbations(df, L, lower_bound, upper_bound)
        for column in df.columns.difference(["timeStart", "SrcIP"]):
            F_ssa = SSA(df[column].values, L)
            components = F_ssa.components_to_df(max_components)
            for i in range(max_components):
                if lower_bound <= hurst(components[f"F{i}"]) <= upper_bound:
                    break
            df[column] = F_ssa.reconstruct(list(range(i)))
        os.makedirs(os.path.dirname(targetFilename), exist_ok=True)
        df.to_csv(targetFilename, index=False)
        # del df, F_ssa, components
        # gc.collect()
    except Exception as e:
        print("Error: " + filename, end=" ")
        print(e)
    return

if __name__ == "__main__":
    pool_size = 6

    pool = Pool(pool_size)

    for filename in glob(Path('interval1/200702111400/*/*').__str__()):
        pool.apply_async(job, (filename,))

    pool.close()
    pool.join()