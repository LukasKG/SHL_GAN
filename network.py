# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import warnings

from log import log

import numpy as np
import preprocessing as pp

P_PATH = 'pic/'
os.makedirs("pic", exist_ok=True)

M_PATH = 'models/'
os.makedirs("models", exist_ok=True)

# File format for vector graphics
FILE_FORMAT_V = '.pdf'

# File format for pixel graphics
FILE_FORMAT_P = '.png'

def save_fig(params,name,fig):
    if params['name'][-1] == '/':
        path = P_PATH+params['name']+name
    else:
        path = P_PATH+params['name']+'_'+name
    os.makedirs(path.rsplit('/', 1)[0], exist_ok=True)
    fig.savefig( path+FILE_FORMAT_V, dpi=300 )
    fig.savefig( path+FILE_FORMAT_P, dpi=300 )

# -------------------
#  Generator
# -------------------

class Generator_01(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Generator_01, self).__init__()
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.tanh = nn.Tanh()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        out = self.tanh(out)
        return out


# -------------------
#  Discriminator
# -------------------    

class Discriminator_01(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Discriminator_01, self).__init__()
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.sig = nn.Sigmoid()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        out = self.sig(out)
        return out


# -------------------
#  Classifier
# -------------------

class Classifier_01(nn.Module):
    ''' Gumbel Softmax (Discrete output is default) '''
    def __init__(self, input_size, hidden_size, num_classes, hard=True):
        super(Classifier_01, self).__init__()
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, num_classes)
        self.hard = hard
        
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        out = F.gumbel_softmax(out, tau=1, hard=self.hard, eps=1e-10, dim=1)
        return out


# -------------------
#  Support Functions
# -------------------

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)

def new_G(run,params,input_size,hidden_size,output_size):
    if params['G_no'] == 1:
        G = Generator_01(input_size, hidden_size, output_size)
    # elif params['G_no'] == 2:
    #     G = Generator_02(input_size, hidden_size, output_size)
    # elif params['G_no'] == 3:
    #     G = Generator_03(input_size, hidden_size, output_size)
    else:
        log("No model Generator_%s"%str(params['G_no']).zfill(2),name=params['log_name'],error=True)
        return None
    
    G.apply(weights_init_normal)
    log("Created new generator.",name=params['log_name'])
    # save_Model(get_string_name(params['name'],run,'G'),G)
    return activate_CUDA(G)

def new_D(run,params,input_size,hidden_size):
    if params['D_no'] == 1:
        D = Discriminator_01(input_size, hidden_size)
    # elif params['D_no'] == 2:
    #     D = Discriminator_02(input_size, hidden_size)
    # elif params['D_no'] == 3:
    #     D = Discriminator_03(input_size, hidden_size)
    else:
        log("No model Discriminator_%s"%str(params['D_no']).zfill(2),name=params['log_name'],error=True)
        return None
    
    D.apply(weights_init_normal)
    log("Created new discriminator.",name=params['log_name'])
    # save_Model(get_string_name(params['name'],run,'D'),D)
    return activate_CUDA(D)

def new_C(run,params,input_size,hidden_size,num_classes):
    if params['C_no'] == 1:
        C = Classifier_01(input_size, hidden_size, num_classes)
    # elif params['C_no'] == 2:
    #     C = Classifier_02(input_size, hidden_size, num_classes)
    # elif params['C_no'] == 3:
    #     C = Classifier_03(input_size, hidden_size, num_classes)
    else:
        log("No model Classifier_%s"%str(params['C_no']).zfill(2),name=params['log_name'],error=True)
        return None
    
    C.apply(weights_init_normal)
    log("Created new classifier.",name=params['log_name'])
    # save_Model(get_string_name(params['name'],run,'C'),C)
    return activate_CUDA(C)

# -------------------
#  Save/Load Networks
# -------------------

def get_string_name(name,run,model):
    if model not in ['G','D','C','R']:
        log("Invalid model type \"%s\""%model,error=True)
        return None
    return '%s_R%d_%s'%(name,run,model)

def save_GAN(name,run,G,D,C):
    save_Model(get_string_name(name,run,'G'),G)
    save_Model(get_string_name(name,run,'D'),D)
    save_Model(get_string_name(name,run,'C'),C)

def save_Ref(name,run,R):
    save_Model(get_string_name(name,run,'R'),R)

def save_Model(name,model):
    PATH = M_PATH+name+'.pt'
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        torch.save(model, PATH)
#    log("Saved model "+name)
    
def load_Ref(run,params):
    input_size, output_size = pp.get_size(params)
    
    # Load Classifier
    R = load_Model(get_string_name(params['name'],run,'R'),params)
    if R is None:
        R = new_C(run, params, input_size=input_size, hidden_size=128, num_classes=output_size)
        
    return R
    
def load_GAN(run,params):
    input_size, output_size = pp.get_size(params)
    
    # Load Generator
    G = load_Model(get_string_name(params['name'],run,'G'),params)
    if G is None:
        G = new_G(run, params, input_size=params['noise_shape']+output_size, hidden_size=256, output_size=input_size)
        
    # Load Discriminator
    D = load_Model(get_string_name(params['name'],run,'D'),params)
    if D is None:
        D = new_D(run, params, input_size=input_size+output_size, hidden_size=128)
        
    # Load Classifier
    C = load_Model(get_string_name(params['name'],run,'C'),params)
    if C is None:
        C = new_C(run, params, input_size=input_size, hidden_size=128, num_classes=output_size)
        
    return G, D, C
    
def load_Model(name,params):
    PATH = M_PATH+name+'.pt'
    
    if not os.path.isfile(PATH):
        log("Model \"%s\" does not exist."%PATH,error=False,name=params['log_name'])
        return None
    
    model = torch.load(PATH)
    model.eval()
    
    log("Loaded model %s."%name,name=params['log_name'])
    
    return activate_CUDA(model)

def activate_CUDA(model):
    if torch.cuda.is_available():
        model.cuda()
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    return model.to(device)

# -------------------
#  Clear Model
# -------------------

def clear(name):
    for fname in os.listdir(M_PATH):
        if fname.startswith(name):
            os.remove(os.path.join(M_PATH, fname))
    log("CLEARED MODEL \""+name+"\"",save=False)

# -------------------
#  Save/Load Parameters
# -------------------
        
def save_Parameter(params):
    PATH = M_PATH+params['name']+'_params.pt'
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        torch.save(params, PATH)
#    log("Saved parameters of model "+name)
    
def load_Parameter(name):
    PATH = M_PATH+name+'_params.pt'
    if not os.path.isfile(PATH):
        # log("Could not find parameters for model \"%s\""%name)
        return None
    return torch.load(PATH)

# -------------------
#  Save/Load Distribution Differences
# -------------------

def save_G_Diff(params,mat):
    PATH = M_PATH+params['name']+'_diff_G.pt'
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        torch.save({'diff_G':mat}, PATH)
        
def load_G_Diff(params):
    PATH = M_PATH+params['name']+'_diff_G.pt'
    mat = np.zeros((params['runs'],int(params['epochs']/params['save_step'])+1,len(params['label'])))
    
    if not os.path.isfile(PATH):
        log("Could not find G differences for model \"%s\""%params['name'],name=params['log_name'])
    else:
        diff = torch.load(PATH)
        mat = fit_array(mat,diff['diff_G'])
        log("Loaded G differences for model \"%s\""%params['name'],name=params['log_name'])
    return mat

# -------------------
#  Save/Load Accuracy
# -------------------

def save_R_Acc(params,mat_R):
    PATH = M_PATH+params['name']+'_acc_R.pt'
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        torch.save({'mat_R':mat_R}, PATH)
#    log("Saved accuracies of model "+name)
    
def load_R_Acc(params):
    PATH = M_PATH+params['name']+'_acc_R.pt'
    mat_R = np.zeros((params['runs'],int(params['epochs']/params['save_step'])+1))
    mat_R[:,0] = 1.0/pp.get_size(params)[1]

    if not os.path.isfile(PATH):
        log("Could not find accuracies for model \"%s\""%params['name'],name=params['log_name'])
    else:
        acc = torch.load(PATH)
        mat_R = fit_array(mat_R,acc['mat_R'])
        log("Loaded accuracies for model \"%s\""%params['name'],name=params['log_name'])
    return mat_R

def save_Acc(params,mat_G,mat_D,mat_C):
    PATH = M_PATH+params['name']+'_acc.pt'
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        torch.save({'mat_G':mat_G,'mat_D':mat_D,'mat_C':mat_C}, PATH)
#    log("Saved accuracies of model "+name)
    
def load_Acc(params):
    PATH = M_PATH+params['name']+'_acc.pt'
    mat_G = np.zeros((params['runs'],int(params['epochs']/params['save_step'])+1))
    mat_G[:,0] = 0.5
    mat_D = np.zeros((params['runs'],int(params['epochs']/params['save_step'])+1))
    mat_D[:,0] = 0.5
    mat_C = np.zeros((params['runs'],int(params['epochs']/params['save_step'])+1))
    mat_C[:,0] = 1.0/pp.get_size(params)[1]

    if not os.path.isfile(PATH):
        log("Could not find accuracies for model \"%s\""%params['name'],name=params['log_name'])
    else:
        acc = torch.load(PATH)
        mat_G = fit_array(mat_G,acc['mat_G'])
        mat_D = fit_array(mat_D,acc['mat_D'])
        mat_C = fit_array(mat_C,acc['mat_C'])
        log("Loaded accuracies for model \"%s\""%params['name'],name=params['log_name'])
    return mat_G, mat_D, mat_C

def fit_array(target,source):
    ''' Fit the content of array 'source' into array 'target' '''
    if source.shape[1]>target.shape[1]:
        source = source[:,:target.shape[1]]
    if source.shape[0]>target.shape[0]:
        source = source[:target.shape[0]]
    target[:source.shape[0],:source.shape[1]] = source
    return target

if __name__ == "__main__":
    print("PyTorch version:",torch.__version__)
    print("      GPU Count:",torch.cuda.device_count())
    print(" Cuda available:",torch.cuda.is_available())
    print("  cudnn enabled:",torch.backends.cudnn.enabled)