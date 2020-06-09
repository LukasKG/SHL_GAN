# -*- coding: utf-8 -*-
from scipy.stats import kurtosis, skew

import numpy as np

# Zero Crossing Count
def zcc(data,*args,**kwargs):
    return np.nonzero(np.diff(data > 0))[0].shape[0]

# Mean Crossing Count
def mcc(data,*args,**kwargs):
    return np.nonzero(np.diff((data-np.mean(data)) > 0))[0].shape[0]

FXdict = {
  "mean": (np.mean,None),
  "std": (np.std,None),
  "median": (np.median,None),
  "zcr": (zcc,None),
  "mcr": (mcc,None),
  "kurtosis": (kurtosis,None),
  "skew": (skew,None)
}

def extract_features(stack,FX_list):
    # Number of extracted features
    FX_size = len(FX_list)
    
    # Number of output channels
    CH_size = len(stack)
    
    # Number of samples
    SA_size = stack[0].shape[0]
    
    data = np.empty((SA_size,CH_size*FX_size),dtype=np.float64)
    
    for nn in range(SA_size):
        for cc in range(CH_size):
            win = stack[cc][nn]
            for ff,FX_name in enumerate(FX_list):
                
                # Check for NaN values and replace with median
                if np.isnan(win.min()):
                    median = np.nanmedian(win)
                    for idx in np.argwhere(np.isnan(win)):
                        win[idx] = median
                      
                # Load Feature function
                FX, FXOpts = FXdict[FX_name]
                
                # Compute the feature
                fx = FX(win,FXOpts)

                # Update the output vector
                data[nn,cc*FX_size+ff] = fx
    
    return data