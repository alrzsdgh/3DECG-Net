# %% [code]
# %% [code]

import random
from tqdm import tqdm
import pandas as pd
import numpy as np
from scipy import signal
from scipy import ndimage


random.seed(42)

def data_preparation(inp_dataset):
    ''' 
    inp_dataset [dict]: a dictionary containing the data (sample number and information
                            key-value pairs) to be split note that keys have to be ordered 
                            integers starting from 1
                            example: {'1':[first_sample], '2':[second_sample], ...}
                            
    output: 
    '''

    tmp_lst = list(range(len(inp_dataset)))   
    ds = []
    for idx in tqdm(tmp_lst):
        ds.append(inp_dataset[str(idx)])
    return ds


def label_creator(inp_dataset, desired_labels):
    '''   
    inp_dataset [list]: a list of the samples and labels
                        example: [[first_signal, first_labels], [second_signals, second_labels],..]
                        
    desired_labels [dict]: a dictionary containing arrhythmias name and code as key-value pairs
                        example: {'AF':'164889003', 'IAVB':'270492004',..}
    '''
    tmp_dict = dict()
    for i in desired_labels.keys():
        tmp_dict[f'{i}'] = []
    label = pd.DataFrame(tmp_dict)
    for i , v in tqdm(enumerate(inp_dataset)):
        tmp_arr = np.zeros((len(desired_labels),))
        for k, l in enumerate(desired_labels.values()):
            if l in v[1][0]:
                tmp_arr[k] = 1
        label.loc[i] = tmp_arr
    return label



def signal_cutter(inp_data, d_length = 1000):
    tmp_signal = []
    for i in tqdm(inp_data):
        tmp_signal.append(i[0][:,:d_length])
    return np.array(tmp_signal)



def signal_filtering(inp, srate = 50):
    tmp_lst = []
    for i in inp:
        p1 = signal.sosfilt(signal.butter(1, 0.5, fs=srate, output='sos',btype='highpass'), i)
        #p2 = signal.sosfilt(signal.butter(1, 0.5, fs=srate, output='sos'), p1)
        tmp_lst.append(p1)
    return np.array(tmp_lst)


def data_filtering(inp_data):
    tmp_lst = []
    for i in tqdm(inp_data):
        tmp_lst.append(signal_filtering(i))
    filtered_ds = np.array(tmp_lst)
    return filtered_ds



def embed_time_series(series, dimension = 2):
    embedded = np.column_stack([series[i:len(series) - dimension + i + 1] for i in range(dimension)])
    return embedded


def calculate_recurrence_matrix(data, threshold = 0.1):
    # Calculate pairwise Euclidean distances
    differences = data[:, np.newaxis, :] - data
    distances = np.linalg.norm(differences, axis=2)
    #recurrence_matrix = (distances <= threshold).astype(int)            
    return distances

def RP_3D(inp_patient):
    tmp_lst = []
    for signal in inp_patient:
        embedded_data = embed_time_series(signal)
        recurrence_matrix = calculate_recurrence_matrix(embedded_data)
        tmp_lst.append(recurrence_matrix)
    out = np.array(tmp_lst)
    m1, m2, m3 = np.shape(out)
    return np.reshape(out, (m1, m2, m3, 1))


def resampling(inp_data, desired_points = 100):
    data = []
    for i in tqdm(inp_data):
        a = signal.resample(i, desired_points, t=None, axis=1, window=None, domain='time')
        data.append(a)
    return np.array(data)


def resize_volume(img, desired_depth = 45, desired_width = 45, desired_height = 45):
    """Resize across z-axis"""
    # Get current depth
    current_height = img.shape[1]
    current_width = img.shape[2]
    current_depth = img.shape[0]
    # Compute depth factor
    depth = current_depth / desired_depth
    width = current_width / desired_width
    height = current_height / desired_height
    depth_factor = 1 / depth
    width_factor = 1 / width
    height_factor = 1 / height
    # Resize across z-axis
    img = ndimage.zoom(img, (depth_factor, height_factor, width_factor, 1))
    return img

def RP_final_data(inp_data):
    data = []
    for i in tqdm(inp_data):
        rp = RP_3D(i)
        res = resize_volume(rp)
        data.append(res)
    return np.array(data)


def including_data(inp_labels, ds, test_size = 1126, n_seed= 42):
    tmp_dict = dict()
    for i in inp_labels.keys():
        tmp_dict[f'{i}'] = []
    in_labels = pd.DataFrame(tmp_dict)
    
    for idx, col in tqdm(enumerate(inp_labels.values)):
        if not (col == np.zeros((7,))).all():
            in_labels.loc[idx] = col
    
    in_lst = list(in_labels.index)
    random.seed(n_seed)
    random.shuffle(in_lst)
    test_idc = in_lst[:test_size]
    train_idc = in_lst[test_size:]
    
    x_train = ds[train_idc]
    y_train = inp_labels.iloc[train_idc]

    x_test = ds[test_idc]
    y_test = inp_labels.iloc[test_idc]
    
    return x_train, y_train, x_test, y_test

# %% [code]