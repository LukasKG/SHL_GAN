# -*- coding: utf-8 -*-
from sklearn import preprocessing

import numpy as np
import matplotlib.pyplot as plt

from network import save_fig

import data_source as ds
import GAN

params = GAN.get_params(name='Loc_analysis',log_name='Loc_analysis',FX_sel='basic')
features = ds.FEATURES[params['FX_sel']]
locations = ['bag','hand','hips','torso']

def load_data(params,dataset):
    X, _ = ds.load_data(params,dataset)
    return X

def get_data(dset):
    data = []

    for loc in locations:
        params['location'] = loc
        X = load_data(params,dset)
        data.append([np.mean(X,axis=0),np.std(X,axis=0)])
    
    params['location'] = 'test'    
    X = load_data(params,'test')
    data.append([np.mean(X,axis=0),np.std(X,axis=0)])
    
    return np.array(data)

def normalize(data):
    ''' x normalized = (x – x minimum) / (x maximum – x minimum) '''
    
    feat = data.shape[2]
    mode = data.shape[1]
    loc = data.shape[0]
    
    data_new = np.zeros_like(data)
    
    for y in range(mode):
        for z in range(feat):
            x_min = data[:,y,z].min()
            x_max = data[:,y,z].max()
            data_new[:,y,z] = (data[:,y,z]-x_min)/((x_max-x_min))
    return data_new

def standardize(data):
    ''' x standardized = (x – u) / o '''
    
    feat = data.shape[2]
    mode = data.shape[1]
    loc = data.shape[0]
    
    data_new = np.zeros_like(data)
    
    for y in range(mode):
        for z in range(feat):
            u = data[:,y,z].mean()
            o = data[:,y,z].std()
            if o == 0.0:
                data_new[:,y,z] = np.zeros_like(data[:,y,z])
                print(data[:,y,z],u,o)
            else:
                data_new[:,y,z] = (data[:,y,z]-u)/o
    return data_new 

def plot(data,name):
    x = range(1,data.shape[2]+1)

    for sub in data:
        plt.errorbar(x, sub[0], sub[1], linestyle='None', marker='x',alpha=0.8)

    plt.legend(locations+['target'])
    plt.xlabel('Feature')
    plt.ylabel('Value')

    plt.xlim(0,data.shape[2]+1)

    plt.grid()
    fig = plt.gcf()
    save_fig(params,name,fig)
    
def calc_dis(data):
    tar = data[len(locations),0]
    for i,loc in enumerate(locations):
        X = data[i,0]
        diff = np.sum((tar-X)**2)**(0.5)
        print(" - diff:",loc,'=',diff)
        
def run(dset):
    data = get_data(dset)
    
    data_norm = normalize(data)
    data_stand = standardize(data)
    
    plot(data_norm,dset+'_normalized')
    plot(data_stand,dset+'_standardized')
    
    print("Normalize")
    calc_dis(data_norm)
    print("\nStandardize")
    calc_dis(data_stand)
    
if __name__ == "__main__":
    run('train')
    run('validation')