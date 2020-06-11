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

DEFAULT_PARAMS = {
        'name'            : "Missing_Name",
        'start_run'       : 0,
        'start_epoch'     : 0,
        'runs'            : 1,
        'epochs'          : 1000,
        'save_step'       : 100,
        'log_name'        : 'log',
        
        'dset_L'          : 'validation',
        'dset_U'          : 'validation',
        'dset_V'          : None,
        'ratio_L'         : 1.0,
        'ratio_U'         : 1.0,  
        'ratio_V'         : 1.0, 
        
        'FX_sel'          : 'basic',
        'location'        : 'hips',
        'prediction'      : True,
        
        'oversampling'    : True,
        'batch_size'      : 128,
        'noise_shape'     : 100,

        'G_no'            : 1,
        'D_no'            : 1,
        'C_no'            : 1,
        
        'G_label_sample'  : True,
        'G_label_factor'  : 1,
        'G_calc_dis'      : False,
        'C_basic_train'   : True,
        'R_active'        : True,
        
        'GLR'             : 0.00125,
        'GB1'             : 0.2,
        'GB2'             : 0.999,
        'DLR'             : 0.001125,
        'DB1'             : 0.7,
        'DB2'             : 0.999,
        'CLR'             : 0.0025,
        'CB1'             : 0.99,
        'CB2'             : 0.999,
        }

# Load params and overwrite missing ones with default values
# Priority:
# 1. Given parameters
# 2. Saved parameters
# 3. Default parameters
def get_params(**kwargs):
    given = locals()['kwargs']
    saved = network.load_Parameter(given.get('name','missingNo'))
    if saved is None:
        saved = DEFAULT_PARAMS
    params = {}
    for key in DEFAULT_PARAMS:
        val = given.get(key,None)
        if val is None:
            val = saved.get(key,None)
            if val is None:
                val = DEFAULT_PARAMS.get(key,None)
        params[key] = val
    params['label'] = pp.get_labels()
    return params
        

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
    
    
def train_GAN(params):
    
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
    G_Loss = torch.nn.BCELoss()
    D_Loss = torch.nn.BCELoss()
    C_Loss = torch.nn.BCELoss()
    
    if cuda:
        G_Loss.cuda()
        D_Loss.cuda()
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
    XV  ... Validation data
    XL  ... Labelled data
    XU  ... Unlabelled data
    '''    
    
    dset_L = params['dset_L']
    dset_U = params['dset_U']
    dset_V = params['dset_V']
    
    if dset_L == dset_U:
        X, Y = pp.get_data(params,dset_L)
        XTL, XTU, YTL, YTU = pp.split_data(X,Y)
    else:
        XTL, YTL = pp.get_data(params,dset_L)
        XTU, YTU = pp.get_data(params,dset_U)
    
    if dset_V is None:
        XV, YV = XTU, YTU
    else:
        XV, YV = pp.get_data(params,dset_V)
    
    XTL = pp.scale_minmax(XTL)
    XTL, YTL = pp.get_tensor(XTL, YTL)
    
    XTU = pp.scale_minmax(XTU)
    XTU, YTU = pp.get_tensor(XTU, YTU)
    
    XV = pp.scale_minmax(XV)
    XV, YV = pp.get_tensor(XV, YV)
    
    # -------------------
    #  Load accuracy
    # -------------------

    mat_accuracy_G, mat_accuracy_D, mat_accuracy_C = network.load_Acc(params)

    if(params['R_active']):
        mat_accuracy_R = network.load_R_Acc(params)
        
    # -------------------
    #  Final prediction
    # -------------------  
    
    if(params['prediction']):
        print(XTU.shape[0])
        Y_pred = torch.zeros(XTU.shape[0],8)
        
    # -------------------
    #  Start Training
    # -------------------
    
    for run in range(params['runs']):
        
        XF = None
        YF = None
        PF = None
        RF = None
        
        # -------------------
        #  Labelled Data
        # -------------------
        
        XL, YL = XTL, YTL
        
        if params['ratio_L'] < 1.0:
            XL, YL = pp.select_random(XL,YL,params['ratio_L'])
            log("Selected %s of training samples."%( format(params['ratio_L'],'0.2f') ),name=params['log_name'])
        
        count_L = YL.shape[0]
        log("Number of labelled samples = %d."%( count_L ),name=params['log_name'])
        if params['oversampling']:
            XL, YL = pp.over_sampling(params, XL, YL)
            log("Oversampling: created %d new samples."%( XL.shape[0]-count_L ),name=params['log_name'])
        
        dataloader = pp.get_dataloader(params, XL, YL)
        
                
        # -------------------
        #  Unlabelled Data
        # -------------------
        
        XU, YU = XTU, YTU
        
        if params['ratio_U'] < 1.0:
            XU, YU = pp.select_random(XU,YU,params['ratio_U'])
            log("Selected %s of validation samples."%( format(params['ratio_U'],'0.2f') ),name=params['log_name'])
        
        log("Number of unlabelled samples = %d."%( XU.shape[0] ),name=params['log_name'])

        iter_UL = pp.get_perm_dataloader(params, XU, YU)
        
        # -------------------
        #  Networks
        # -------------------
        
        G, D, C = network.load_GAN(run,params)
        
        if(params['R_active']):
            R = network.load_Ref(run,params)
        
        # -------------------
        #  Optimizers
        # -------------------
        
        optimizer_G = torch.optim.Adam(G.parameters(), lr=params['GLR'], betas=(params['GB1'], params['GB2']))
        optimizer_D = torch.optim.Adam(D.parameters(), lr=params['DLR'], betas=(params['DB1'], params['DB2']))
        optimizer_C = torch.optim.Adam(C.parameters(), lr=params['CLR'], betas=(params['CB1'], params['CB2']))
        
        if(params['R_active']):
            optimizer_R = torch.optim.Adam(R.parameters(), lr=params['CLR'], betas=(params['CB1'], params['CB2']))
        
        # -------------------
        #  Training
        # -------------------
        
        if run >= params['start_run']:
            for epoch in range(params['epochs']):
                
                # Jump to start epoch
                if run == params['start_run']:
                    if epoch < params['start_epoch']:
                        continue
                
                running_loss_G = 0.0
                running_loss_D = 0.0
                running_loss_C = 0.0
                
                
                """
                      X1, P1      - Labelled Data,      predicted Labels (C)                             | Regular training of classifier
                W1 = (X1, Y1), A1 - Labelled Data,      actual Labels,        predicted Authenticity (D) | Real samples
                W2 = (X2, Y2), A2 - Unlabelled Data,    predicted Labels (C), predicted Authenticity (D) | Real data with fake labels
                W3 = (X3, Y3), A3 - Synthetic Data (G), actual Labels,        predicted Authenticity (D) | Fake data with real labels
                W4 = (X4, Y4), A4 - Unlabbeled Data,    predicted Labels (C), predicted Authenticity (D) | Fake positive to prevent overfitting
                      XV, YV,  PV - Validation Data,    actual Labels,        predicted Labels (C)       | Validation samples
                  R1, F2, F3,  R4 - Real/Fake Labels
                """
                for i, data in enumerate(dataloader, 1):
                    
                    loss_G = []
                    loss_D = []
                    loss_C = []
                    
                    # -------------------
                    #  Train the classifier on real samples
                    # -------------------
                    X1, Y1 = data
                    W1 = torch.cat((X1,Y1),dim=1)
                    R1 = floatTensor(W1.shape[0], 1).fill_(1.0)
                    
                    if params['C_basic_train']:
                        optimizer_C.zero_grad()
                        P1 = C(X1)
                        loss = C_Loss(P1, Y1)
                        loss_C.append(loss)
                        loss.backward()
                        optimizer_C.step()
                    
                    if params['R_active']:
                        optimizer_R.zero_grad()
                        PR = R(X1)
                        loss = C_Loss(PR, Y1)
                        loss.backward()
                        optimizer_R.step()
                        
                    # -------------------
                    #  Train the discriminator to label real samples
                    # -------------------
                    optimizer_D.zero_grad()
                    A1 = D(W1)
                    loss = D_Loss(A1, R1)
                    loss_D.append(loss)
                    loss.backward()
                    optimizer_D.step()
                    
                    # -------------------
                    #  Classify unlabelled data
                    # -------------------
                    optimizer_C.zero_grad()
                    X2, _ = iter_UL.get_next()
                    Y2 = C(X2)
                    W2 = torch.cat((X2,Y2),dim=1)

                    # -------------------
                    #  Train the classifier to label unlabelled samples
                    # -------------------
                    A2 = D(W2)
                    R2 = floatTensor(W2.shape[0], 1).fill_(1.0)
                    loss = C_Loss(A2, R2)
                    loss_C.append(loss)
                    loss.backward()
                    optimizer_C.step()
                    
                    # -------------------
                    #  Train the discriminator to label predicted samples
                    # -------------------
                    optimizer_D.zero_grad()
                    A2 = D(W2.detach())
                    F2 = floatTensor(W2.shape[0], 1).fill_(0.0)
                    loss = D_Loss(A2, F2)
                    loss_D.append(loss)
                    loss.backward()
                    optimizer_D.step()
                    
                    # -------------------
                    #  Train the discriminator to label fake positive samples
                    # -------------------
                    X4, _ = iter_UL.get_next()
                    Y4 = C(X4)
                    W4 = torch.cat((X4,Y4),dim=1)
                    
                    optimizer_D.zero_grad()
                    A4 = D(W4)
                    R4 = floatTensor(W4.shape[0], 1).fill_(1.0)
                    loss = D_Loss(A4, R4)
                    loss_D.append(loss)
                    loss.backward()
                    optimizer_D.step()
                    
                    # -------------------
                    #  Create Synthetic Data
                    # -------------------     
                    optimizer_G.zero_grad()
                    if params['G_label_sample']:
                        # Selected Labels from a uniform distribution of available labels
                        Y3 = floatTensor(pp.get_one_hot_labels(params=params,num=Y1.shape[0]*params['G_label_factor']))
                    else:
                        # Select labels from current training batch
                        Y3 = torch.cat(([Y1 for _ in range(params['G_label_factor'])]),dim=0)
                    
                    Z = floatTensor(np.random.normal(0, 1, (Y3.shape[0], params['noise_shape'])))
                    I3 = torch.cat((Z,Y3),dim=1)
                    X3 = G(I3)
                    W3 = torch.cat((X3,Y3),dim=1)
                    
                    # -------------------
                    #  Train the generator to fool the discriminator
                    # -------------------
                    A3 = D(W3)
                    R3 = floatTensor(W3.shape[0], 1).fill_(1.0)
                    loss = G_Loss(A3, R3)
                    loss_G.append(loss)
                    loss.backward()
                    optimizer_G.step()
                    
                    # -------------------
                    #  Train the discriminator to label synthetic samples
                    # -------------------
                    optimizer_D.zero_grad()
                    A3 = D(W3.detach())
                    F3 = floatTensor(W3.shape[0], 1).fill_(0.0)
                    loss = D_Loss(A3, F3)
                    loss_D.append(loss)
                    loss.backward()
                    optimizer_D.step()
                    
                    # -------------------
                    #  Calculate overall loss
                    # -------------------
                    running_loss_G += np.mean([loss.item() for loss in loss_G])
                    running_loss_D += np.mean([loss.item() for loss in loss_D])
                    running_loss_C += np.mean([loss.item() for loss in loss_C])
                
                # -------------------
                #  Post Epoch
                # -------------------
                
                logString = "[Run %d/%d] [Epoch %d/%d] [G loss: %f] [D loss: %f] [C loss: %f]"%(run+1, params['runs'], epoch+1, params['epochs'], running_loss_G/(i), running_loss_D/(i), running_loss_C/(i))
                log(logString,save=False,name=params['log_name'])
                
                if (epoch+1)%params['save_step'] == 0:
                    # log("~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~|",save=False,name=params['log_name'])
                    idx = run,int(epoch/params['save_step'])+1
                    
                    # Predict labels
                    PV = C(XV)
                    
                    if params['R_active']:
                        PR = R(XV)
                        mat_accuracy_R[idx] = get_accuracy(PR, YV)
                        network.save_Ref(params['name'],run,R)
                        network.save_R_Acc(params, mat_accuracy_R)
                    
                    # Generate Synthetic Data
                    Z = floatTensor(np.random.normal(0, 1, (YV.shape[0], params['noise_shape'])))
                    IV = torch.cat((Z,YV),dim=1)
                    XG = G(IV)
                    
                    
                    # Estimate Discriminator Accuracy
                    WV1 = torch.cat((XV,YV),dim=1)
                    WV2 = torch.cat((XV,PV),dim=1)
                    WV3 = torch.cat((XG,YV),dim=1)
                    RV1 = floatTensor(WV1.shape[0],1).fill_(1.0)
                    FV2 = floatTensor(WV2.shape[0],1).fill_(0.0)
                    FV3 = floatTensor(WV3.shape[0],1).fill_(0.0)
                    
                    AV1 = D(WV1)
                    AV2 = D(WV2)
                    AV3 = D(WV3)
                    
                    acc_D_real = get_accuracy_binary(AV1,RV1)
                    acc_D_vs_C = get_accuracy_binary(AV2,FV2)
                    acc_D_vs_G = get_accuracy_binary(AV3,FV3)
                    acc_D = .5*acc_D_real + .25*acc_D_vs_G + .25*acc_D_vs_C
                    mat_accuracy_D[idx] = acc_D
                
                    acc_C_real = get_accuracy(PV, YV)
                    acc_C_vs_D = 1.0 - acc_D_vs_C
                    acc_C = .5*acc_C_real + .5*acc_C_vs_D
                    mat_accuracy_C[idx] = acc_C_real
                
                    acc_G = 1.0 - acc_D_vs_G
                    mat_accuracy_G[idx] = acc_G
                
                    logString = "[Run %d/%d] [Epoch %d/%d] [G acc: %f] [D acc: %f | vs Real: %f | vs G: %f | vs C: %f] [C acc: %f | vs Real: %f | vs D: %f]"%(run+1, params['runs'], epoch+1, params['epochs'], acc_G, acc_D, acc_D_real, acc_D_vs_G, acc_D_vs_C, acc_C, acc_C_real, acc_C_vs_D)
                    log(logString,save=True,name=params['log_name']) 
                    
                    network.save_GAN(params['name'],run,G,D,C)
                    params['start_epoch'] = epoch+1
                    network.save_Parameter(params)
                    network.save_Acc(params, mat_accuracy_G, mat_accuracy_D, mat_accuracy_C)
                                
                    # log("~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~|",save=False,name=params['log_name'])
            
            # End of Training Run
            params['start_run'] = run+1
            params['start_epoch'] = 0
            network.save_Parameter(params)
            
        # -------------------
        #  Post Run
        # -------------------
                
        # Generate Synthetic Data
        Z = floatTensor(np.random.normal(0, 1, (YV.shape[0], params['noise_shape'])))
        IV = torch.cat((Z,YV),dim=1)
        XG = G(IV).detach()
        
        # Classify Validation data
        PC = C(XV).detach()
        if params['R_active']:
            if RF == None:
                RF = R(XV)
            else:
                RF = torch.cat((RF, R(XV).detach()), 0)
            
        if XF == None:
            XF = XG
            YF = YV
            PF = PC
        else:
            XF = torch.cat((XF, XG), 0)
            YF = torch.cat((YF, YV), 0)
            PF = torch.cat((PF, PC), 0)
        
        # -------------------
        #  Final prediction
        # -------------------  
        
        if(params['prediction']):
            C.hard = False
            Y_pred += C(XTU).cpu().detach()
            C.hard = True
            
    # -------------------
    #  Post Training
    # -------------------

    timeline = np.arange(0,params['epochs']+1,params['save_step'])
    
    # -------------------
    #  Plot Accuracy
    # -------------------
    
    acc_G = np.mean(mat_accuracy_G,axis=0)
    acc_D = np.mean(mat_accuracy_D,axis=0)
    acc_C = np.mean(mat_accuracy_C,axis=0)
    if params['R_active']:
        acc_R = np.mean(mat_accuracy_R,axis=0)
        
    fig, ax = plt.subplots()    
    
    legend = []  
    cmap = plt.get_cmap('gnuplot')
    indices = np.linspace(0, cmap.N, 7)
    colors = [cmap(int(i)) for i in indices]

    ax.plot(timeline,acc_C,c=colors[0],linestyle='solid')
    legend.append("Accuracy $A_C$")
    
    ax.plot(timeline,acc_D,c=colors[1],linestyle='dashed')
    legend.append("Accuracy $A_D$")
    
    ax.plot(timeline,acc_G,c=colors[2],linestyle='dotted')
    legend.append("Accuracy $A_G$")
    
    Y_max = 1.15
    if params['R_active']:
        ax.plot(timeline,acc_R,c=colors[3],linestyle='dashdot')
        legend.append("Accuracy $A_R$")
        
        perf = (acc_C-acc_R)/acc_R
        ax.plot(timeline,perf+1,c=colors[4],linestyle='solid')
        legend.append("Performance $P_C$")
    
    ax.set_xlim(0.0,params['epochs'])
    ax.set_ylim(0.0,Y_max)
    
    ax.legend(legend)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
        
    ax.grid()
    save_fig(params,'eval',fig)

    # -------------------
    #  Compare Classifier to Baseline
    # -------------------
   
    if params['R_active']:
        maxC = np.argmax(acc_C, axis=0)
        bestC = acc_C[maxC]
        maxR = np.argmax(acc_R, axis=0)
        bestR = acc_R[maxR]
        log(' - Peak Accuracy: C: %s after %d epochs | R: %s after %d epochs | Inc: %s'%(
            format((bestC),'0.4f'),timeline[maxC],
            format((bestR),'0.4f'),timeline[maxR],
            format((bestC-bestR)/bestR,'0.4f')),name='results')
        
        Y_max = max(Y_max,max(perf+1)+0.025)
        
        maxP = np.argmax(perf, axis=0)
        log(' - Hightest $P_C$: %s after %d epochs.'%(format((perf[maxP]),'0.4f'),timeline[maxP]),name='results')
        
        adva = np.zeros_like(acc_C)
        for i,v1 in enumerate(acc_C):
            for j,v2 in enumerate(acc_R):
                if v2>=v1:
                    adva[i] = j-i
                    break
                
        maxA = np.argmax(adva, axis=0)
        log(' - Biggest Advantage: %d epochs after %d epochs.'%(adva[maxA]*params['save_step'],timeline[maxA]),name='results')  
    
    # -------------------
    #  Generate Confusion Matrix
    # -------------------
      
    YF = pp.one_hot_to_labels(params,YF)
    PF = pp.one_hot_to_labels(params,PF)
    
    con_mat = confusion_matrix(YF, PF, labels=None, sample_weight=None, normalize='true')
    plot_confusion_matrix(con_mat,params,name='C',title='Confusion matrix')
    
    if params['R_active']:
        RF = pp.one_hot_to_labels(params,RF)
        con_mat = confusion_matrix(YF, RF, labels=None, sample_weight=None, normalize='true')
        plot_confusion_matrix(con_mat,params,name='R',title='Confusion matrix')
    
    # -------------------
    #  Final prediction
    # -------------------  
    
    if(params['prediction']):
        pred = torch.argmax(Y_pred,axis=1)        
        f = open(network.S_PATH+params['name']+'_predictions.txt', "w")
        for y in pred:
            f.write(' '.join(['%.6f'%(float(y.item()+1))]*500)+'\n')
        f.close()
    
    # -------------------
    #  Log Results
    # -------------------
    
    log(" - "+params['name']+": [C acc: %f] [D acc: %f] [G acc: %f]"%(acc_C[-1],acc_D[-1],acc_G[-1]),name='results')
    
if __name__ == "__main__":
    import main
    main.test()