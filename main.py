# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
plt.ioff()

from log import log,clear
from GAN import get_params
from pre_train import train_C

import network
import excel
import GAN

def train(params,data=None):
    """
    Parameters:
        name            -       Name of the network
        start_run       -       At which run training is started
        runs            -       Number of runs
        epochs          -       Number of epochs
        save_step       -       At which epoch period the log is saved
        log_name        -       Name of the logfile
        
        dset_L          -       Dataset used as labelled data
        dset_U          -       Dataset used as unlabelled data
        dset_V          -       Dataset used as validation, if None, dset_U is used
        ratio_L         -       Ratio of labelled samples used
        ratio_U         -       Ratio of unlabelled samples used
        ratio_V         -       Ratio of validation samples used
        
        FX_sel          -       List of features to apply to the sliding window
        Location        -       Which body locations (bag, hand, hips, torso, or all)
        prediction      -       True: creates a file containing the prediction for the validation data
        pretrain        -       If given: Name of the pretrained model to be loaded
        
        oversampling    -       True: oversample all minority classes
        batch_size      -       Number of samples per batch
        noise_shape     -       Size of random noise Z
        
        G_no            -       Model number of the new generator
        D_no            -       Model number of the new discriminator
        C_no            -       Model number of the new classifier
        
        G_label_sample  -       True: randomly sample input labels for G | False: use current sample batch as input
        G_label_factor  -       Size factor of the input for G in relation to current batch
        G_calc_dis      -       True: calculate and plot the distance between real and generated Data [NOT IMPLEMENTED]
        C_basic_train   -       True: The classifier is trained on real data | False: the classifier is only trained against the discriminator
        R_active        -       True: a reference classifier is used as baseline
        
        GLR             -       Generator learning rate
        GB1             -       Generator decay rate for first moment estimates
        GB2             -       Generator decay rate for second-moment estimates
        DLR             -       Discriminator learning rate
        DB1             -       Discriminator decay rate for first moment estimates
        DB2             -       Discriminator decay rate for second-moment estimates
        CLR             -       Classifier learning rate
        CB1             -       Classifier decay rate for first moment estimates
        CB2             -       Classifier decay rate for second-moment estimates
    """
    
    GAN.train_GAN(params)

def cross_train(def_params,param_1,param_1_lst,param_2,param_2_lst,last_bmark=0):
    
    clear('results')
    results = np.zeros((len(param_1_lst),len(param_2_lst),3))
    
    count = 1
    for x,p1 in enumerate(param_1_lst):
        for y,p2 in enumerate(param_2_lst):
            
            c_name = def_params['name'] + '_' + str(count).zfill(3)
            params = network.load_Parameter(c_name)
            if params is None:
                params = def_params.copy()
                params['name'] = c_name
                params[param_1] = p1
                params[param_2] = p2
            else:
                params = get_params(**params)
                
            # Continue at the last benchmark
            if count > last_bmark:       
                log('Benchmark %s: %s = %s | %s = %s'%(c_name,param_1,str(p1),param_2,str(p2)),name='results')
                train(params)
            
            mat_accuracy_G, mat_accuracy_D, mat_accuracy_C = network.load_Acc(params)
            
            acc_G = np.mean(mat_accuracy_G,axis=0)
            acc_D = np.mean(mat_accuracy_D,axis=0)
            acc_C = np.mean(mat_accuracy_C,axis=0)

            results[x,y,1] = acc_G[-1]
            results[x,y,2] = acc_D[-1]
            results[x,y,0] = acc_C[-1]
            
            count += 1
            
    excel.save(def_params,param_1,param_1_lst,param_2,param_2_lst,results)

def test_cross():
    name = 'test_cross'
    
    params = get_params(
            name = name,
            FX_sel = 'basic',
            location = 'hips',
            dset_L = 'validation',
            dset_U = 'validation',
        
            runs=3,
            epochs=6,
            save_step=3,

            oversampling = False,

            G_label_sample = True,
            G_label_factor = 1,
            C_basic_train = True ,
            R_active = False,
            
            G_no = 1,
            D_no = 1,
            C_no = 1,
            
            ratio_L = 0.05,
            ratio_U = 0.33,
            
            log_name = 'log')
    network.save_Parameter(params)
 
    param_1 = 'GLR'
    param_1_lst = [0.0005,0.001,0.0015]
    param_2 = 'GB1'
    param_2_lst = [0.3,0.7,0.9,0.99]
    
    log('Crosstrain param %s against param %s.'%(param_1,param_2),name=params['log_name'])
    log('List 1: [%s]'%(' '.join( str(v) for v in param_1_lst )),name=params['log_name'])
    log('List 2: [%s]'%(' '.join( str(v) for v in param_2_lst )),name=params['log_name'])
    log('Total runs: %d'%(len(param_1_lst)*len(param_2_lst)),name=params['log_name'])
    
    cross_train(params,param_1,param_1_lst,param_2,param_2_lst,0)

def cross_params(net,lst_1,lst_2,last_bmark=0):
    name = 'cross_'+net
    
    params = get_params(
            name = name,
            FX_sel = 'basic',
            location = 'hips',
            
            dset_L = 'validation',
            dset_U = 'validation',
            dset_V = None,
            ratio_L = 0.5,
            ratio_U = 0.5,
        
            runs=5,
            epochs=150,
            save_step=15,

            oversampling = True,

            G_label_sample = True,
            G_label_factor = 1,
            C_basic_train = True ,
            R_active = False,
            
            G_no = 1,
            D_no = 1,
            C_no = 1,
    
            
            log_name = 'log')
    network.save_Parameter(params)
 
    param_1 = net+'LR'
    param_1_lst = lst_1
    param_2 = net+'B1'
    param_2_lst = lst_2
    
    log('Crosstrain param %s against param %s.'%(param_1,param_2),name=params['log_name'])
    log('List 1: [%s]'%(' '.join( str(v) for v in param_1_lst )),name=params['log_name'])
    log('List 2: [%s]'%(' '.join( str(v) for v in param_2_lst )),name=params['log_name'])
    log('Total runs: %d'%(len(param_1_lst)*len(param_2_lst)),name=params['log_name'])
    
    cross_train(params,param_1,param_1_lst,param_2,param_2_lst,last_bmark)

def bmark_LR_G(last_bmark=0):
    """benchmark different Learning Rates and Beta1 Decays for the Generator Optimiser"""
    net = 'G'
    lst_1 = [0.00125,0.0013,0.00135]
    lst_2 = [0.15,0.2,0.25]
    
    cross_params(net,lst_1,lst_2,last_bmark)

def bmark_LR_D(last_bmark=0):
    """benchmark different Learning Rates and Beta1 Decays for the Discriminator Optimiser"""
    net = 'D'
    lst_1 = [0.011125,0.01125,0.01137,0.015]
    lst_2 = [0.725,0.75,0.775]
    
    cross_params(net,lst_1,lst_2,last_bmark)

def bmark_LR_C(last_bmark=0):
    """benchmark different Learning Rates and Beta1 Decays for the Classifier Optimiser"""
    net = 'C'
    lst_1 = [0.002125,0.00225,0.00237]
    lst_2 = [0.7,0.8,0.9]
    
    cross_params(net,lst_1,lst_2,last_bmark)

def test():
    name = "test"    
    network.clear(name)
    
    params = get_params(
            name = name,
            FX_sel = 'basic',
            location = 'hips',

            dset_L = 'train',
            dset_U = 'validation',
            dset_V = 'validation',
            ratio_L = 1,
            ratio_U = 1,
            ratio_V = 1,
        
            pretrain = 'final',
        
            runs=1,
            epochs=10,
            save_step=2,

            oversampling = True,

            G_label_sample = True,
            G_label_factor = 1,
            C_basic_train = True ,
            R_active = True,
            
            G_no = 1,
            D_no = 1,
            C_no = 1,
            
            log_name = 'log')
    
 
    train(params=params)
    GAN.get_prediction_accuracy(params)

def basic():
    name = "Val_Val"    
    
    params = get_params(
            name = name,
            FX_sel = 'basic',
            location = 'hips',
            
            dset_L = 'validation',
            dset_U = 'test',
            dset_V = 'validation',
            ratio_L = 1.0,
            ratio_U = 1.0,
            ratio_V = 1.0,
        
            pretrain = 'pretrain',
        
            runs=5,
            epochs=50,
            save_step=2,

            oversampling = True,

            G_label_sample = True,
            G_label_factor = 1,
            C_basic_train = True ,
            R_active = False,
            
            G_no = 1,
            D_no = 1,
            C_no = 1, 
            
            log_name = 'log')
 
    train(params=params)

def pretrain():
    name = 'pretrain'
    
    params = get_params(
        name = name,
        FX_sel = 'basic',
        location = 'hips',
        
        dset_L = 'train',
        dset_V = 'validation',
        ratio_L = 1.00,
        ratio_V = 1.00,
    
        runs=20,
        epochs=25,
        save_step=5,

        oversampling = True,

        C_no = 1,

        log_name = 'log')
 
    train_C(params=params)
    
def main():
    # test()
    # test_cross()
    
    # bmark_LR_G(last_bmark=0)
    # bmark_LR_D(last_bmark=0)
    bmark_LR_C(last_bmark=0)
    
    # pretrain()
    basic()
    

if __name__ == "__main__":
    main()
