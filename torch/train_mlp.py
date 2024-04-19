import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from battery_data import getDischargeMultipleBatteries
def get_data_tensor(data_dict, max_idx_to_use, max_size):
    inputs_list = []
    target_list = []

    for k, v in data_dict.items():
        for i, d in enumerate(v[1, :max_idx_to_use]):
            prep_inp = np.full(max_size, np.nan)
            prep_target = np.full(max_size, np.nan)
            prep_inp[:len(d)] = d   # Current Sequence
            prep_target[:len(v[0, :][i])] = v[0, :][i]  # Voltage sequence

            inputs_list.append(prep_inp)
            target_list.append(prep_target)

    inputs_array = np.array(inputs_list)
    target_array = np.array(target_list)

   
    inputs_tensor = torch.tensor(inputs_array, dtype=torch.float32)
    target_tensor = torch.tensor(target_array, dtype=torch.float32)

    return inputs_tensor, target_tensor

# Load battery data
data_RW = getDischargeMultipleBatteries()
max_idx_to_use = 3
max_size = np.max([v[0, 0].shape[0] for k, v in data_RW.items()])
dt = np.diff(data_RW[1][2, 0])[1]
# Get data tensors
inputs_tensor, target_tensor = get_data_tensor(data_RW, max_idx_to_use, max_size)


BATCH_SIZE = inputs_tensor.shape[0]


dataset = TensorDataset(inputs_tensor.unsqueeze(-1), target_tensor.unsqueeze(-1))
train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)