import matplotlib.pyplot as plt
import pandas as pd
import spear as sp
import numpy as np
from pathlib import Path  
import time
import math
import os

DATA_PATH    = 'D:\SEVIR Data\data'
CATALOG_PATH = 'D:\SEVIR Data/CATALOG.csv'

# Read catalog
catalog = pd.read_csv(CATALOG_PATH,parse_dates=['time_utc'],low_memory=False)

# Desired image types
img_types = set(['vil'])

# Group by event id, and filter to only events that have all desired img_types
events = catalog.groupby('id').filter(lambda x: img_types.issubset(set(x['img_type']))).groupby('id')
event_ids = list(events.groups.keys())
print('Found %d events matching' % len(event_ids),img_types)

def createParameters(threshold, nSTD):
    data_list = []
    try:
        folder_path = "D:\\Final Dataset\\threshold_" + str(threshold) + "_nSTD_" + str(nSTD)
        os.makedirs(folder_path)
    except:
        print("File already exists")
    df_ignored = []
    start_time = time.time()
    features = ['EventID', 
                'FileName',
                'SemiMajorAxisLength', 
                'SemiMinorAxisLength',
                'Area',
                'AxisRatio',
                'EllipseAngle', 
                'COMX', 
                'COMY', 
                'Average',
                'Intensity',
                'VelocityX',
                'VelocityY',
                'AngularVelocity']
    flag = True
    
    
    
# CHANGE FOR LOOP BELOW TO COMMENTED OUT LINE!!!!!



    for event_index in range(0, len(event_ids)):
#     for event_index in range(0, 100):
        sample_event = events.get_group(event_ids[event_index])
        if(sample_event.size > 21):
            sample_event = sample_event.loc[sample_event['img_type'] == 'vil']
            if(sample_event.size > 21):
                sample_event = pd.DataFrame(sample_event.iloc[0,:])
                sample_event = sample_event.T    
        event_Number = event_ids[event_index]
    #     print(sample_event)
        percent_missing = sample_event['pct_missing'].item()
        print(event_index, event_Number, percent_missing)

        if percent_missing > 20:
            df_ignored.append([event_Number, file_name])
            flag = False

        file_name = sample_event['file_name'].item()
        vil = sp.read_data(sample_event, 'vil')
        parameterVector = []

        if flag:
            for frame in range(0, vil.shape[2]):
                image = vil[:,:,frame]
                try:
                    major, minor, area, axisRatio, angle, com, average, intensity = sp.findStormParams(image, threshold, nSTD)
                except:
                    df_ignored.append([event_Number, file_name])
                    print('This file not valid')
                    flag = False
                    break

                comX = com[0]
                comY = com[1]
                
                params = [event_Number, file_name, major, minor, area, axisRatio, angle, comX, comY, average, intensity, 0.0, 0.0, 0.0]
                parameterVector.append(params)

        if flag:
            df = pd.DataFrame(parameterVector, columns = features)
            velocityX = sp.findCOMVelocity(df.loc[:,'COMX'])
            velocityY = sp.findCOMVelocity(df.loc[:,'COMY'])
            angularVelocity = sp.findAngularVelocity(df.loc[:,'EllipseAngle'])
            df.loc[1:48,'VelocityX'] = velocityX
            df.loc[1:48,'VelocityY'] = velocityY
#             df.loc[48,'VelocityX'] = df.loc[47,'VelocityX']
#             df.loc[48,'VelocityY'] = df.loc[47,'VelocityY']
            df.loc[0,'VelocityX'] = 0
            df.loc[0,'VelocityY'] = 0
            df.loc[0:48,'VelocityMagnitude'] = np.sqrt((df['VelocityX'] ** 2) + (df['VelocityY'] ** 2))
#             df.loc[48,'VelocityMagnitude'] = df.loc[47,'VelocityMagnitude']
            df.loc[1:48,'AngularVelocity'] = angularVelocity 
#             df.loc[48,'AngularVelocity'] = df.loc[47,'AngularVelocity']
            df.loc[0,'AngularVelocity'] = 0
            paramFileName = event_Number + '_Params.csv'
            cols = df.columns.tolist()
            cols = cols[:-2] + [cols[-1]] + [cols[-2]]
            df=df[cols]
            df = df.drop(index=0)
            filepath = folder_path + "\\" + paramFileName
#             df.to_csv(filepath, index=False)
            data_list.append(df.to_numpy())
        flag = True
    print("--- %s seconds ---" % (time.time() - start_time))
#     print(data_list[0])
    np_data = np.array(data_list)
    np_path = "D:\\Final Dataset\\threshold_" + str(threshold) + "_nSTD_" + str(nSTD) + "\\np_array_object"
    try:
        os.makedirs(np_path)
    except:
        print("File already exists")    
    np_path = np_path + "\\np_array.npy"
    np.save(np_path, np_data)
    np_path = "D:\\Final Dataset\\threshold_" + str(threshold) + "_nSTD_" + str(nSTD) + "\\np_array_no_text.npy"
    np.save(np_path, np_data[:,:,2:])
    # First return is data np array. Second return is data no array without the first 2 columns, which are text with event/file names
    return np_data, np_data[:,:,2:]


def removeZeroEvents(dataset):
    new_set = []
    for event in dataset:
        if not(np.any(event[:, 0:2] < 3)):
            new_set.append(event)
    new_set = np.array(new_set)
    return new_set



def normalizeDataset(train, test):
    stacked_train = np.array(np.vstack(train))
    stacked_train = stacked_train.astype(np.float64)
    means = np.mean(stacked_train, axis=0)
    stds =  np.std(stacked_train, axis=0)
    normalized_train = (stacked_train - means) / stds
    train_return = normalized_train.reshape(-1,49,13)
    
    stacked_test = np.array(np.vstack(test))
    stacked_test = stacked_test.astype(np.float64)
    normalized_test = (stacked_test - means) / stds
    test_return = normalized_test.reshape(-1,49,13)
    return train_return, test_return



def createXY(dataset, input_length, horizon_length):
    X = []
    Y = []
    for sample in dataset:
        for i in range(len(sample) - input_length - horizon_length + 1):
            x = torch.tensor(sample[i:i+input_length], dtype=torch.float32)
            y = torch.tensor(sample[i+input_length:i+input_length+horizon_length], dtype=torch.float32)
#             x_padded = pad_sequence(x, max_input_len, 0)
#             y_padded = pad_sequence(y, max_horizon_len, 0)
            X.append(x)
            Y.append(y)
    return X, Y





def createModelInputs(norm_train, norm_test, input_lengths, horizon_lengths):
    train_dataset = {}
    for input_length in input_lengths:
        for horizon_length in horizon_lengths:
#             print("Crunching numbers for Input: " + str(input_length) + " Horizon: " + str(horizon_length))
            cur_x, cur_y = createXY(norm_train, input_length, horizon_length)
            key = (input_length, horizon_length)  # Create a key for the combination
            train_dataset[key] = {"X": [], "Y": []}
            train_dataset[key]["X"].append(cur_x)
            train_dataset[key]["Y"].append(cur_y)

    # Create merged X Y for train dataset
    X_all_Train = []
    Y_all_Train = []
    total = 0
    for input_len, output_horizon in product(input_lengths, horizon_lengths):
        # Get the data for the current (input_len, output_horizon) pair
        X = train_dataset[(input_len, output_horizon)]["X"][0]
        Y = train_dataset[(input_len, output_horizon)]["Y"][0]
        X_all_Train = X_all_Train + X
        Y_all_Train = Y_all_Train + Y

    train_x_lengths = []
    train_y_lengths = []

    for input_length in input_lengths:
        for horizon_length in horizon_lengths:
            key = (input_length, horizon_length)  # Create a key for the combination
            train_x_lengths = train_x_lengths + [input_length]*len(train_dataset[key]["X"][0])
            train_y_lengths = train_y_lengths + [horizon_length]*len(train_dataset[key]["Y"][0])
            
    X_all_Train = []
    Y_all_Train = []
    total = 0
    for input_len, output_horizon in product(input_lengths, horizon_lengths):
        # Get the data for the current (input_len, output_horizon) pair
        X = train_dataset[(input_len, output_horizon)]["X"][0]
        Y = train_dataset[(input_len, output_horizon)]["Y"][0]
        X_all_Train = X_all_Train + X
        Y_all_Train = Y_all_Train + Y

    train_x_lengths = []
    train_y_lengths = []

    for input_length in input_lengths:
        for horizon_length in horizon_lengths:
            key = (input_length, horizon_length)  # Create a key for the combination
            train_x_lengths = train_x_lengths + [input_length]*len(train_dataset[key]["X"][0])
            train_y_lengths = train_y_lengths + [horizon_length]*len(train_dataset[key]["Y"][0])

    test_dataset = {}
    for input_length in input_lengths:
        for horizon_length in horizon_lengths:
#             print("Crunching numbers for Input: " + str(input_length) + " Horizon: " + str(horizon_length))
            cur_x, cur_y = createXY(norm_test, input_length, horizon_length)
            key = (input_length, horizon_length)  # Create a key for the combination
            test_dataset[key] = {"X": [], "Y": []}
            test_dataset[key]["X"].append(cur_x)
            test_dataset[key]["Y"].append(cur_y)
            
    return X_all_Train, Y_all_Train, train_x_lengths, train_y_lengths, test_dataset


# -----------------------------------------------------------------------------
# PYTORCH DATA LOADING FUNCTIONS
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from itertools import product
from torch.nn.utils.rnn import pad_sequence
import numpy as np
from itertools import product  # Import product for Cartesian product
import pandas as pd
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
# from torch.nn.utils.rnn import pad_sequence
import torch.optim as optim
import torch


class SeqDataset(Dataset):
    def __init__(self, X_data, Y_data, X_lengths, Y_lengths):
        self.X_data = X_data
        self.Y_data = Y_data
        self.X_lengths = X_lengths
        self.Y_lengths = Y_lengths

    def __len__(self):
        return len(self.X_data)

    def __getitem__(self, idx):
        X = self.X_data[idx]
        Y = self.Y_data[idx]
        X_len = self.X_lengths[idx]
        Y_len = self.Y_lengths[idx]
        
        return X, Y, X_len, Y_len
    
# Custom collate_fn to pad X and unpack Y sequences
def collate_fn(batch):
    # Sort by length in descending order (for packing)
    batch.sort(key=lambda x: x[2], reverse=True)
    
    X, Y, X_lengths, Y_lengths = zip(*batch)
    
    # Pad X sequences
    X_padded = pad_sequence(X, batch_first=True, padding_value=0)
    
    # Stack Y sequences (no padding here, since Y is unpadded)
    Y_padded = pad_sequence(Y, batch_first=True, padding_value=0)
    
    # Convert lists to tensors
    X_lengths = torch.tensor(X_lengths)
    Y_lengths = torch.tensor(Y_lengths)

    return X_padded, Y_padded, X_lengths, Y_lengths


def createTrainDataLoader(train_set, test_set, horizon_lengths, input_lengths):
    X_all_Train, Y_all_Train, train_x_lengths, train_y_lengths, test_dataset = createModelInputs(train_set, test_set, input_lengths, horizon_lengths)
    training_torch_dataset = SeqDataset(X_all_Train, Y_all_Train, train_x_lengths, train_y_lengths)
    train_data_loader = DataLoader(training_torch_dataset, batch_size=256, shuffle=True, drop_last=True, collate_fn=lambda batch: collate_fn(batch))
    return train_data_loader, training_torch_dataset, test_dataset
    
    
    