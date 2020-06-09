# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sn

from network import save_fig

def plot_confusion_matrix(cm,params,name='C',title='Confusion Matrix'):
    """
    given a sklearn confusion matrix (cm), make a nice plot

    """
    fig, ax = plt.subplots()
    ax = sn.heatmap(cm, annot=True, fmt='.1%',xticklabels=params['label'], yticklabels=params['label'],cmap = sn.cubehelix_palette(8), vmin=0, vmax=1)
    
    cbar = ax.collections[0].colorbar
    cbar.set_ticks([0, .5, 1])
    cbar.set_ticklabels(['0%', '50%', '100%'])
    
    ax.set_ylabel("Actual")
    ax.set_xlabel("Predicted")
    
    fig.tight_layout()
    save_fig(params,'con_mat_'+name,fig)