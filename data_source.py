# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import os

from log import log
from sliding_window import extract_features, FXdict

DATA_FILES = [
    'Acc_x.txt','Acc_y.txt','Acc_z.txt',
    'Gra_x.txt','Gra_y.txt','Gra_z.txt',
    'Gyr_x.txt','Gyr_y.txt','Gyr_z.txt',
    'LAcc_x.txt','LAcc_y.txt','LAcc_z.txt',
    'Mag_x.txt','Mag_y.txt','Mag_z.txt',
    'Ori_w.txt','Ori_x.txt','Ori_y.txt','Ori_z.txt',
    'Pressure.txt'
    ]

LABELS_SHL = {
        1: "Still",
        2: "Walking",
        3: "Run",
        4: "Bike",
        5: "Car",
        6: "Bus",
        7: "Train",
        8: "Subway",
        }

PATHS = {
    'Challenge_test': 'D:\\data\\SHL_Challenge_2020\\test\\',
    'Challenge_train': 'D:\\data\\SHL_Challenge_2020\\train\\',
    'Challenge_validation': 'D:\\data\\SHL_Challenge_2020\\validation\\',
    'Data_test': 'D:\\data\\SHL_2020_prepared\\test\\',
    'Data_train': 'D:\\data\\SHL_2020_prepared\\train\\',
    'Data_validation': 'D:\\data\\SHL_2020_prepared\\validation\\',
        }

FEATURES = {
    'basic': ['mean','std','mcr','kurtosis','skew'],
    'all': FXdict.keys(),
    }

def load_data(params,dataset):
    '''
    Checks if the selected dataset-location combination is already extracted.
    If not, the according data is loaded, features extracted, and the result stored.
    Then the selected data and - if available - according labels are loaded and returned.

    Parameters
    ----------
    dataset : name of the dataset
    location : location of the sensor
    FX_sel : selection of features

    Returns
    -------
    Data X
    Labels Y (optional, otherwise None)

    '''
    if dataset == 'test':
        location = 'test'
    else:
        location = params['location']
    FX_sel = params['FX_sel']
    
    assert dataset in ['test','train','validation']
    assert location in ['bag','hand','hips','torso','all','test']
    assert (dataset == 'test') == (location == 'test')
    assert FX_sel in ['basic','all']
    
    log("Loading dataset %s.. (Location: %s | FX: %s)"%(dataset,location,FX_sel),name=params['log_name'])
    
    if location == 'all':
        params_tmp = params.copy()
        
        params_tmp['location'] = 'bag'
        X1, Y1 = load_data(params_tmp,dataset)
        
        params_tmp['location'] = 'hand'
        X2, Y2 = load_data(params_tmp,dataset)
        
        params_tmp['location'] = 'hips'
        X3, Y3 = load_data(params_tmp,dataset)
        
        params_tmp['location'] = 'torso'
        X4, Y4 = load_data(params_tmp,dataset)
        
        data = np.concatenate((X1, X2, X3, X4),axis=0)
        label = np.concatenate((Y1, Y2, Y3, Y4),axis=0)
    
    else:
        path = PATHS['Data_'+dataset] + location + '\\' + FX_sel + '\\'
        if not os.path.isfile(path+'data.txt'):
            log("Generating dataset..",name=params['log_name'])
            generate_data(dataset,location,FX_sel)
        
        data = pd.read_csv(path+'data.txt',header=None).to_numpy()
        
        if os.path.isfile(path+'label.txt'):
            label = pd.read_csv(path+'label.txt',header=None).to_numpy()
        else:
            label = None
    
    log("Dataset %s (%s) loaded."%(dataset,location),name=params['log_name'])
    return data, label

def read_data(dataset,location,channel_selection=range(len(DATA_FILES))):
    src_path = PATHS['Challenge_'+dataset] + location + '\\'
    stack = []
    for i,filename in enumerate(DATA_FILES):
        if i in channel_selection:
            X = pd.read_csv(src_path+filename,sep=' ',header=None).to_numpy()
            stack.append(X)
    return stack

def generate_data(dataset,location,FX_sel):
    src_path = PATHS['Challenge_'+dataset] + location + '\\'
    tar_path = PATHS['Data_'+dataset] + location + '\\' + FX_sel + '\\'
    os.makedirs(tar_path, exist_ok=True)
    
    stack = read_data(dataset,location)
    data = extract_features(stack,FEATURES[FX_sel])
    df = pd.DataFrame(data=data, index=None, columns=None)
    df.to_csv(tar_path+'data.txt',header=False,index=False)
    
    if os.path.isfile(src_path+'Label.txt'):
        labels = pd.read_csv(src_path+'Label.txt',sep=' ',header=None).to_numpy()
        labels = labels[:,0]
        df = pd.DataFrame(data=labels, index=None, columns=None)
        df.to_csv(tar_path+'label.txt',header=False,index=False)

def read_prediction(params,src_path):
    if os.path.isfile(src_path):
        return pd.read_csv(src_path,sep=' ',header=None).to_numpy()[:,0]
    else:
        log("Can't find predictions for model %s."%(params['name']),name=params['log_name'])
        return None
        
def load_FX(FX_sel):
    from GAN import get_params
    params = get_params(FX_sel=FX_sel)
    
    params['location'] = 'all'
    data, label = load_data(params,'validation')
    print('Loaded!')
    print(" Data:",data.shape)
    print("  NaN:",np.count_nonzero(np.isnan(data)))
    print("Label:",label.shape)
    print("----------------------")
    
    params['location'] = 'bag'
    data, label = load_data(params,'train')
    print('Loaded!')
    print(" Data:",data.shape)
    print("  NaN:",np.count_nonzero(np.isnan(data)))
    print("Label:",label.shape)
    print("----------------------")
    
    params['location'] = 'hand'
    data, label = load_data(params,'train')
    print('Loaded!')
    print(" Data:",data.shape)
    print("  NaN:",np.count_nonzero(np.isnan(data)))
    print("Label:",label.shape)
    print("----------------------")
    
    params['location'] = 'hips'
    data, label = load_data(params,'train')
    print('Loaded!')
    print(" Data:",data.shape)
    print("  NaN:",np.count_nonzero(np.isnan(data)))
    print("Label:",label.shape)
    print("----------------------")
    
    params['location'] = 'torso'
    data, label = load_data(params,'train')
    print('Loaded!')
    print(" Data:",data.shape)
    print("  NaN:",np.count_nonzero(np.isnan(data)))
    print("Label:",label.shape)
    print("----------------------")
    
    params['location'] = 'test'
    data, label = load_data(params,'test')
    print('Loaded!')
    print(" Data:",data.shape)
    print("  NaN:",np.count_nonzero(np.isnan(data)))
    print("----------------------")
    
if __name__ == "__main__":
    load_FX('basic')
    load_FX('all')
