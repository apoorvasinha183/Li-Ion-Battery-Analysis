import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from battery_data import BatteryDataFile, getDischargeMultipleBatteries, DATA_PATH, BATTERY_FILES
import time

#from BatteryRNNCell_mlp import BatteryRNN
from model import get_model
#DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE = torch.device("cpu")

###### FOR REFERENCE : DATA INGESTION STARTS HERE ##########
data_RW = getDischargeMultipleBatteries(varnames=['voltage', 'current', 'time'], discharge_type='discharge (random walk)')

max_size = 0
num_seq = 10

inputs = []
inputs_time = []
target = []

for k,rw_data in data_RW.items():
    # skip batteries RW 9 to 12 for now (RW does not got to EOD)
    if k>8:
        continue
    # rw_data = data_RW[1]
    time = np.hstack([rw_data[2][i] for i in range(len(rw_data[2]))])
    time = time - time[0]
    current_inputs = np.hstack([rw_data[1][i] for i in range(len(rw_data[1]))])
    voltage_target = np.hstack([rw_data[0][i] for i in range(len(rw_data[0]))])

    last_idx = 0
    seq_durations = np.diff([0]+list(np.argwhere(np.diff(time)>10)[:,0]+1))
    
    for curr_duration in seq_durations[:num_seq]:
        if curr_duration>max_size:
            max_size = curr_duration
        curr_idx = last_idx + curr_duration
        inputs.append(current_inputs[last_idx:curr_idx])
        inputs_time.append(time[last_idx:curr_idx])
        target.append(voltage_target[last_idx:curr_idx])
        last_idx = curr_idx
inputs_orig = inputs.copy()
targets_orig = target.copy()
inputs_orig = np.vstack(inputs_orig)[:,:,np.newaxis]
targets_orig = np.vstack(targets_orig)
# add nan to end of seq to have all seq in same size
for i in range(len(inputs)):
    prep_inputs = np.full(max_size, np.nan)
    prep_target = np.full(max_size, np.nan)
    prep_inputs_time = np.full(max_size, np.nan)
    prep_inputs[:len(inputs[i])] = inputs[i]
    prep_target[:len(target[i])] = target[i]
    prep_inputs_time[:len(inputs_time[i])] = inputs_time[i]
    inputs[i] = prep_inputs
    target[i] = prep_target
    inputs_time[i] = prep_inputs_time

inputs = np.vstack(inputs)[:,:,np.newaxis]
target = np.vstack(target)
inputs_time = np.vstack(inputs_time)

time_window_size = inputs.shape[1]
BATCH_SIZE = inputs.shape[0]
dt = np.diff(data_RW[1][2,0])[1]


# move timesteps with earlier EOD
EOD = 3.2

inputs_shiffed = inputs.copy()
target_shiffed = target.copy()
reach_EOD = np.ones(BATCH_SIZE, dtype=int) * time_window_size
for row in np.argwhere((target<EOD) | (np.isnan(target))):
    if reach_EOD[row[0]]>row[1]:
        reach_EOD[row[0]]=row[1]
        row_1 = row[1]
        if ~np.isnan(inputs[row[0],:,0][row[1]]):
            row_1 = row[1] + 1
        inputs_shiffed[row[0],:,0] = np.zeros(time_window_size)
        inputs_shiffed[row[0],:,0][time_window_size-row_1:] = inputs[row[0],:,0][:row_1]
        target_shiffed[row[0]] = np.ones(time_window_size) * target[row[0]][0]
        target_shiffed[row[0]][time_window_size-row_1:] = target[row[0]][:row_1]

###### FOR REFERENCE : DATA INGESTION ENDS HERE ##########



###### FOR REFERENCE : MODEL LOADING STARTS HERE ##########

mlp = get_model(dt=dt, mlp=True, share_q_r=False, stateful=True).to(DEVICE)
weights_path = 'torch_train/mlp_trained_weights.pth'
mlp_p_weights = torch.load(weights_path)

with torch.no_grad():
    mlp.cell.qMax.copy_(mlp_p_weights['cell.qMax']) 
    mlp.cell.Ro.copy_(mlp_p_weights['cell.Ro']) 
    mlp.cell.MLPp[1].weight.copy_(mlp_p_weights["cell.MLPp.1.weight"])
    mlp.cell.MLPp[1].bias.copy_(mlp_p_weights['cell.MLPp.1.bias'])
    mlp.cell.MLPp[3].weight.copy_(mlp_p_weights['cell.MLPp.3.weight'])
    mlp.cell.MLPp[3].bias.copy_(mlp_p_weights['cell.MLPp.3.bias'])
    mlp.cell.MLPp[5].weight.copy_(mlp_p_weights['cell.MLPp.5.weight'])
    mlp.cell.MLPp[5].bias.copy_(mlp_p_weights['cell.MLPp.5.bias'])

###### FOR REFERENCE : MODEL LOADING ENDS HERE ##########

######## Validation is done here ##########
mlp.eval()
# Time for the test set
X = inputs_orig[::10,:,:]
Y = targets_orig[::10,:,np.newaxis]
X_tensor = torch.from_numpy(X).to(DEVICE)
Y_tensor = torch.from_numpy(Y).to(DEVICE)
with torch.no_grad():
    pred = mlp(X_tensor).cpu().numpy()

#plt.plot(X, Y, color='gray')
print("Predictions have shape ",pred.shape)
for i in range(X.shape[0]):
    fig = plt.figure(figsize=(10, 5))
    ax1 = fig.add_subplot(111)
    ax1.set_xlabel('time')
    ax1.set_ylabel('Voltage [V]')
    ax2 = ax1.twinx()
    ax1.plot(pred[i,:,0], linestyle='dashed', color='red', label='Voltage Prediction')
    ax1.plot(Y_tensor[i,:,0], color='blue', label='Voltage Measured')
    ax1.legend()
    ax2.plot(X_tensor[i,:,0], color='purple', label='Current Measured', alpha=0.5)
    ax2.legend()
    ax2.set_ylabel('Current [I]')
    ax2.yaxis.label.set_color('purple')
    ax2.spines["right"].set_edgecolor('purple')
    ax2.tick_params(axis='y', colors='purple')

    plt.savefig(f'figures/predictionvsreality_random_walk_unshiffed{i}.png')
    plt.show()
######## Validation is done here ##########


######## Validation is done here ##########
mlp.eval()
# Time for the test set
X = inputs_shiffed[::10,:,:]
Y = target_shiffed[::10,:,np.newaxis]
X_tensor = torch.from_numpy(X).to(DEVICE)
Y_tensor = torch.from_numpy(Y).to(DEVICE)
with torch.no_grad():
    pred = mlp(X_tensor).cpu().numpy()

#plt.plot(X, Y, color='gray')
print("Predictions have shape ",pred.shape)
for i in range(X.shape[0]):
    fig = plt.figure(figsize=(10, 5))
    ax1 = fig.add_subplot(111)
    ax1.set_xlabel('time')
    ax1.set_ylabel('Voltage [V]')
    ax2 = ax1.twinx()
    ax1.plot(pred[i,:,0], linestyle='dashed', color='red', label='Voltage Prediction')
    ax1.plot(Y_tensor[i,:,0], color='blue', label='Voltage Measured')
    ax1.legend()
    ax2.plot(X_tensor[i,:,0], color='purple', label='Current Measured', alpha=0.5)
    ax2.legend()
    ax2.set_ylabel('Current [I]')
    ax2.yaxis.label.set_color('purple')
    ax2.spines["right"].set_edgecolor('purple')
    ax2.tick_params(axis='y', colors='purple')

    plt.savefig(f'figures/predictionvsreality_random_walk_shiffed{i}.png')
    plt.show()
######## Validation is done here ##########