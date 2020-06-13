# -*- coding: utf-8 -*-
from sklearn.metrics import confusion_matrix
from plot_confusion_matrix import plot_confusion_matrix

import numpy as np
import torch
import matplotlib.pyplot as plt
plt.ioff()

from log import log
from network import save_fig

import preprocessing as pp
import network

def get_accuracy(prediction,label):
    _, idx_C = label.max(1)
    _, idx_P = prediction.max(1)
    
    cases = list(label.size())[0]
    correct = list(torch.where(idx_C==idx_P)[0].size())[0]
    
    return correct/cases
   
def get_accuracy_binary(prediction,label):
    cases = list(label.size())[0]
    correct = list(torch.where(prediction.round()==label)[0].size())[0]
    
    return correct/cases
    
    
def train_C(params):
    
    # -------------------
    #  Parameters
    # -------------------
    
    log(str(params),name=params['log_name'])
    
    # # Clear remaining model
    # network.clear(params['name']+'_R'+str(params['start_run']))
    
    # -------------------
    #  CUDA
    # -------------------
    
    cuda = True if torch.cuda.is_available() else False
    C_Loss = torch.nn.BCELoss()
    
    if cuda:
        C_Loss.cuda()
        floatTensor = torch.cuda.FloatTensor
        log("CUDA Training.",name=params['log_name'])
    else:
        floatTensor = torch.FloatTensor
        log("CPU Training.",name=params['log_name'])
    
    
    # -------------------
    #  Data scaling
    # -------------------
    '''
    XTL ... Training data labelled
    XTU ... Training data unlabelled
    
    XL  ... Labelled data
    XU  ... Unlabelled data
    XV  ... Validation data
    '''    
    
    dset_L = params['dset_L']
    dset_V = params['dset_V']

    XTL, YTL = pp.get_data(params,dset_L)
    XV, YV = pp.get_data(params,dset_V)
    
    XTL = pp.scale_minmax(XTL)
    XV = pp.scale_minmax(XV)
    
    if params['ratio_V'] < 1.0:
        XV, YV = pp.select_random(XV,YV,params['ratio_L'])
        log("Selected %s of validation samples."%( format(params['ratio_V'],'0.2f') ),name=params['log_name'])
    XV, YV = pp.get_tensor(XV, YV)
    
    # -------------------
    #  Load accuracy
    # -------------------

    mat_accuracy_C = network.load_R_Acc(params)
        
    # -------------------
    #  Start Training
    # -------------------
    
    YF = None
    PF = None
    
    for run in range(params['runs']):
        
        # -------------------
        #  Training Data
        # -------------------
        
        XL, YL = XTL, YTL
        
        if params['ratio_L'] < 1.0:
            XL, YL = pp.select_random(XL,YL,params['ratio_L'])
            log("Selected %s of labelled samples."%( format(params['ratio_L'],'0.2f') ),name=params['log_name'])
        
        count_L = YL.shape[0]
        log("Number of labelled samples = %d."%( count_L ),name=params['log_name'])
        
        dataloader = pp.get_dataloader(params, XL, YL)
        
        
        C = network.load_Ref(run,params)
        
        # -------------------
        #  Optimizers
        # -------------------
        
        optimizer_C = torch.optim.Adam(C.parameters(), lr=params['CLR'], betas=(params['CB1'], params['CB2']))
        
        # -------------------
        #  Training
        # -------------------
        
        if run >= params['start_run']:
            
            if params['oversampling']:
                XL, YL = pp.over_sampling(params, XL, YL)
                log("Oversampling: created %d new labelled samples."%( XL.shape[0]-count_L ),name=params['log_name'])
            
            for epoch in range(params['epochs']):
                
                # Jump to start epoch
                if run == params['start_run']:
                    if epoch < params['start_epoch']:
                        continue
                
                running_loss_C = 0.0
                
                for i, data in enumerate(dataloader, 1):
                    
                    loss_C = []
                    
                    # -------------------
                    #  Train the classifier on real samples
                    # -------------------
                    X1, Y1 = data
                    optimizer_C.zero_grad()
                    P1 = C(X1)
                    loss = C_Loss(P1, Y1)
                    loss_C.append(loss)
                    loss.backward()
                    optimizer_C.step()
                    

                    # -------------------
                    #  Calculate overall loss
                    # -------------------
                    running_loss_C += np.mean([loss.item() for loss in loss_C])
                
                # -------------------
                #  Post Epoch
                # -------------------
                
                logString = "[Run %d/%d] [Epoch %d/%d] [C loss: %f]"%(run+1, params['runs'], epoch+1, params['epochs'], running_loss_C/(i))
                log(logString,save=False,name=params['log_name'])
                
                if (epoch+1)%params['save_step'] == 0:
                    # log("~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~|",save=False,name=params['log_name'])
                    idx = run,int(epoch/params['save_step'])+1
                    
                    # Predict labels
                    PV = C(XV)

                    acc_C_real = get_accuracy(PV, YV)
                    mat_accuracy_C[idx] = acc_C_real
                
                    logString = "[Run %d/%d] [Epoch %d/%d] [C acc: %f ]"%(run+1, params['runs'], epoch+1, params['epochs'], acc_C_real)
                    log(logString,save=True,name=params['log_name']) 
                    
                    network.save_Ref(params['name'],run,C)
                    network.save_R_Acc(params, mat_accuracy_C)
                    
                    params['start_epoch'] = epoch+1
                    network.save_Parameter(params)
                                
                    # log("~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~|",save=False,name=params['log_name'])
            
            # End of Training Run
            params['start_run'] = run+1
            params['start_epoch'] = 0
            network.save_Parameter(params)
            
        # -------------------
        #  Post Run
        # -------------------
                
        # Classify Validation data
        PC = C(XV).detach()
            
        if YF == None:
            YF = YV
            PF = PC
        else:
            YF = torch.cat((YF, YV), 0)
            PF = torch.cat((PF, PC), 0)
            
    # -------------------
    #  Post Training
    # -------------------

    timeline = np.arange(0,params['epochs']+1,params['save_step'])
    
    # -------------------
    #  Plot Accuracy
    # -------------------
    
    acc_C = np.mean(mat_accuracy_C,axis=0)
        
    fig, ax = plt.subplots()    
    
    legend = []  
    cmap = plt.get_cmap('gnuplot')
    indices = np.linspace(0, cmap.N, 7)
    colors = [cmap(int(i)) for i in indices]

    ax.plot(timeline,acc_C,c=colors[0],linestyle='solid')
    legend.append("Accuracy $A_C$")
    
    ax.set_xlim(0.0,params['epochs'])
    ax.set_ylim(0.0,1.0)
    
    ax.legend(legend)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
        
    ax.grid()
    save_fig(params,'eval',fig)

    # -------------------
    #  Generate Confusion Matrix
    # -------------------
      
    YF = pp.one_hot_to_labels(params,YF)
    PF = pp.one_hot_to_labels(params,PF)
    
    con_mat = confusion_matrix(YF, PF, labels=None, sample_weight=None, normalize='true')
    plot_confusion_matrix(con_mat,params,name='C',title='Confusion matrix')
    
    # -------------------
    #  Log Results
    # -------------------
    
    log(" - "+params['name']+": [C acc: %f]"%(acc_C[-1]),name='results')
    