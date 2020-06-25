# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 17:55:35 2020

@author: lukas
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

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

def plot_spectrum(dset, loc, name):
    params['name'] = '_'.join(['spectrum',dset,loc,''])

    channels = []
    figs = []
    # Only select the first 3 (acceleration) channels
    for i in channel_selection:
        channel = ds.DATA_FILES[i][:-4]
        channels.append(channel)
        
        fig, ax = plt.subplots()
        
        ax.set_title(channel)
        ax.grid()
        
        figs.append([fig,ax])
        
    data = ds.read_data(dset,loc,channel_selection)
    print('Loaded dataset %s (%s)'%(dset,loc))
        
    acc = get_acc(data)
    channel = 'Acceleration'
    
    cmap = plt.get_cmap('gnuplot')
    indices = np.linspace(0, cmap.N, 7)
    colors = [cmap(int(i)) for i in indices]
    plt.rcParams.update({'font.size': 18})
    plt.rcParams.update({'figure.autolayout': True})
    
    fig, ax = plt.subplots()
    ax.set_title(channel)
    ax.grid()
    
    ax.plot(test_acc_mag[1:25],c=colors[0],linestyle='solid')
    ax.plot(get_mean_mag(acc)[1:25],c=colors[1],linestyle='dashed')
    ax.legend(['$test$',name])
    ax.set_yscale('log')
    
    ax.set_xlabel('Hz')
    ax.set_ylabel('Spectral Power')
    ax.set_ylim(10**1,10**2.5)
    
    #plt.tight_layout()
    save_fig(params,channel,fig)
    plt.close("all")
    mpl.rcParams.update(mpl.rcParamsDefault)
    
params = GAN.get_params(name='spectrum',log_name='spectrum')
#channel_selection = range(len(DATA_FILES))
channel_selection = range(3)

test_data = ds.read_data('test','test',channel_selection)
test_fft = []
for X in test_data:
    test_fft.append(get_mean_mag(X))
test_acc = get_acc(test_data)
test_acc_mag = get_mean_mag(test_acc)
    
datasets = ['validation','train']
locations = ['bag','hand','hips','torso']

for dset in datasets:
    for loc in locations:
        plot_spectrum(dset, loc, '$%s_{%s}$'%(dset,loc))