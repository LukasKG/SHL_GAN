# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 17:55:35 2020

@author: lukas
"""

import numpy as np
import matplotlib.pyplot as plt

from network import save_fig

import data_source as ds
import GAN

def get_mean_mag(X):
    vec = X - np.mean(X,axis=1,keepdims=True)
    fft = np.fft.fft(vec,axis=1)
    mag = np.abs(fft)
    mean = np.mean(mag,axis=0) 
    return mean

def get_acc(data):
    acc = np.array(data[:3])
    acc = acc ** 2
    acc = np.sum(acc,axis=0)
    acc = acc ** 0.5
    return acc

def plot_spectrum(dset):
    params['name'] = '_'.join(['spectrum',dset])

    # channels = []
    # figs = []
    # # Only select the first 3 (acceleration) channels
    # for i in channel_selection:
    #     channel = ds.DATA_FILES[i][:-4]
    #     channels.append(channel)
        
    #     fig, ax = plt.subplots()
        
    #     ax.set_title(channel)
    #     ax.grid()
        
    #     figs.append([fig,ax])
    
    legend = ['$test$']
    
    channel = 'Acceleration'
    
    cmap = plt.get_cmap('gnuplot')
    indices = np.linspace(0, cmap.N, 6)
    colors = [cmap(int(i)) for i in indices]
    #plt.rcParams.update({'font.size': 20})
    
    fig, ax = plt.subplots()
    #ax.set_title(channel)
    ax.grid()
    
    ax.plot(test_acc_mag[1:26],c=colors[0],linestyle='solid')
    
    styles = ['dashdot','dashed','dotted',(0, (3, 1, 1, 1, 1, 1))]
    for i,loc in enumerate(locations):
    
        data = ds.read_data(dset,loc,channel_selection)
        print('Loaded dataset %s (%s).'%(dset,loc))
        legend.append('$%s$'%(loc))
            
        acc = get_acc(data)
        ax.plot(get_mean_mag(acc)[1:26],c=colors[1+i],linestyle=styles[i],linewidth=2.0)
    if dset=='train': 
        ax.legend(legend,fontsize=20)
    ax.set_yscale('log')
    
    ax.set_xlabel('Hz',fontsize=20)
    ax.set_ylabel('Spectral Power',fontsize=20)
    ax.set_ylim(10**1,10**2.5)
    
    save_fig(params,channel,fig)
    plt.close("all")
    
params = GAN.get_params(name='spectrum',log_name='spectrum')
#channel_selection = range(len(DATA_FILES))
channel_selection = range(3)

test_data = ds.read_data('test','test',channel_selection)
print('Loaded dataset test.')
test_fft = []
for X in test_data:
    test_fft.append(get_mean_mag(X))
test_acc = get_acc(test_data)
test_acc_mag = get_mean_mag(test_acc)
    
datasets = ['validation','train']
locations = ['bag','hand','hips','torso']

for dset in datasets:
    plot_spectrum(dset)