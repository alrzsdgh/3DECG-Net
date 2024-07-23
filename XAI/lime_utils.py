# %% [code]

import matplotlib.pyplot as plt
import numpy as np
from itertools import combinations

def tp_finder(inp_pred, inp_label, inp_target, inp_arr_dict):
    '''
    inp_pred: the predictions made by the black box model
    inp_label: the labels of test samples
    inp_target: target arrhythmia (for instance 'AF' or 'LBBB')
    inp_arr_dict: a dict of arrhythmias name and codes
    '''
    
    idx = list(inp_arr_dict.keys()).index(inp_target)
    tmp_lst = []
    for i in range(len(inp_label)):
        if (inp_label.values[:,idx][i]==1) & (inp_pred[:,idx][i]==1):
            tmp_lst.append(i)
    return tmp_lst



def lead_replacing(inp_num = 4):
    '''
    inp_num: maximum number of leads that can be set to zero (default = 4)
    '''
    
    tmp1 = np.ones((12,1))
    all_vectors = []
    
    for j in range(1,inp_num+1):
        tmp2 = list(combinations(range(12), j))
        for i in tmp2:
            cv = tmp1.copy()
            cv[list(i)] = 0
            all_vectors.append(cv)
        
    return all_vectors


def weight_bars(inp_v):
    colors = ['g' if e >= 0 else 'r' for e in inp_v]
    plt.bar(range(12), inp_v, width = 0.75, color=colors)
    plt.axis('off')
    plt.axline((0,0),(11,0),color='black')
    plt.show()