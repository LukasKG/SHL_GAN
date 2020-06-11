# -*- coding: utf-8 -*-
from sklearn import preprocessing

import numpy as np
import torch

from log import log

import data_source as ds

def get_labels():
    """ Returns list with unique labels """
    return np.fromiter(ds.LABELS_SHL.keys(), dtype=int)

def get_size(params):
    ''' Returns the input shape and number of outputclasses of a dataset '''
    X = 20*len(ds.FEATURES[params['FX_sel']])
    Y = get_labels().shape[0]
    return [X,Y]

def scale_minmax(X):
    ''' Scale data between -1 and 1 to fit the Generators tanh output '''
    scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1), copy=True)
    scaler.fit(X)
    return scaler.transform(X)

def select_random(X0,Y0,ratio):
    ''' Select random samples based on a given target ratio '''
    ix = np.random.choice(len(X0), size=int(ratio*len(X0)), replace=False)
    X1 = X0[ix]
    if Y0 is None:
        Y1 = None
    else:
        Y1 = Y0[ix]
    return X1, Y1
    

from imblearn.over_sampling import SMOTE
def over_sampling(params,X,Y):
    labels = one_hot_to_labels(params,Y)
    smote = SMOTE(sampling_strategy='not majority',k_neighbors=5)
    data, labels = smote.fit_sample(X, labels)
    return data, labels_to_one_hot(params,labels)

from sklearn.model_selection import train_test_split
def split_data(X,Y):
    return train_test_split(X, Y, test_size=0.5)

def get_one_hot_labels(params,num):
    ''' Turns a list with label indeces into a one-hot label array '''
    labels = np.random.choice(params['label'], size=num, replace=True, p=None)
    return labels_to_one_hot(params,labels)

def labels_to_one_hot(params,labels):
    Y = np.zeros((labels.shape[0],params['label'].shape[0]))
    for i in range(labels.shape[0]):
        j = np.where(params['label']==labels[i])[0][0]
        Y[i,j] = 1
    return Y

def one_hot_to_labels(params,Y):
    if torch.is_tensor(Y):
        Y = Y.detach().cpu().numpy()
    return np.array([params['label'][np.where(oh==max(oh))[0][0]] for oh in Y])

def get_data(params,dataset):
    X, Y = ds.load_data(params,dataset)
    if Y is not None:
        Y = labels_to_one_hot(params,Y)
    return X,Y

def get_tensor(X, Y=None):    
    cuda = True if torch.cuda.is_available() else False
    
    if cuda:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    
    X = torch.from_numpy(X).float().to(device)
    if Y is not None:
        Y = torch.from_numpy(Y).float().to(device)
    
    return X, Y

class Permanent_Dataloader:
    def __init__(self, dataloader):
        self.dataloader = dataloader
        self.iterator = iter(self.dataloader)
    def get_next(self):
        try:
            data = next(self.iterator)
        except StopIteration:
            # StopIteration is thrown if dataset ends
            # reinitialize data loader 
            self.iterator = iter(self.dataloader)
            data = next(self.iterator)
        return data

def get_dataloader(params,X,Y):
    # transform to torch tensors
    if not torch.is_tensor(X):
        X, Y = get_tensor(X,Y)
    
    # create your datset
    if Y is not None:
        dataset = torch.utils.data.TensorDataset(X,Y)
    else:
        dataset = torch.utils.data.TensorDataset(X)
    
    # Configure data loader
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=params['batch_size'],
        shuffle=True,
    )
    
    return dataloader

def get_perm_dataloader(params,X,Y):
    dataloader = get_dataloader(params,X,Y)
    perm_dataloader = Permanent_Dataloader(dataloader)
    return perm_dataloader