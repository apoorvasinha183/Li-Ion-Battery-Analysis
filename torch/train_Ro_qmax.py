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
NUM_EPOCHS = 1000

###### FOR REFERENCE : DATA INGESTION STARTS HERE ##########
data_RW = getDischargeMultipleBatteries(varnames=['voltage', 'current', 'time'], discharge_type='discharge (random walk)')

max_size = 0
num_seq = 4

inputs = []
inputs_time = []
target = []
for k,rw_data in data_RW.items():
    # skip batteries RW 9 to 12 for now (RW does not got to EOD)
    if k>8:
        continue
    # rw_data = data_RW[1]
    time_ = np.hstack([rw_data[2][i] for i in range(len(rw_data[2]))])
    time_ = time_ - time_[0]
    current_inputs = np.hstack([rw_data[1][i] for i in range(len(rw_data[1]))])
    voltage_target = np.hstack([rw_data[0][i] for i in range(len(rw_data[0]))])

    last_idx = 0
    seq_durations = np.diff([0]+list(np.argwhere(np.diff(time_)>10)[:,0]+1))
    
    for curr_duration in seq_durations[:num_seq]:
        if curr_duration>max_size:
            max_size = curr_duration
        curr_idx = last_idx + curr_duration
        inputs.append(current_inputs[last_idx:curr_idx])
        inputs_time.append(time_[last_idx:curr_idx])
        target.append(voltage_target[last_idx:curr_idx])
        last_idx = curr_idx

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

val_idx = np.linspace(0,31,6,dtype=int)
train_idx = [i for i in np.arange(0,32) if i not in val_idx]
###### FOR REFERENCE : DATA INGESTION ENDS HERE ##########





###### FOR REFERENCE : TRAINING STARTS HERE #########
        
# Create the MLP model, optimizer, and criterion
mlp = get_model(dt=dt, mlp=True, share_q_r=False, stateful=True).to(DEVICE)


# load trained weights for MLPp
weights_path = 'torch_train/mlp_trained_weights.pth'
mlp_p_weights = torch.load(weights_path)

with torch.no_grad():
    mlp.cell.MLPp[1].weight.copy_(mlp_p_weights["cell.MLPp.1.weight"])
    mlp.cell.MLPp[1].bias.copy_(mlp_p_weights['cell.MLPp.1.bias'])
    mlp.cell.MLPp[3].weight.copy_(mlp_p_weights['cell.MLPp.3.weight'])
    mlp.cell.MLPp[3].bias.copy_(mlp_p_weights['cell.MLPp.3.bias'])
    mlp.cell.MLPp[5].weight.copy_(mlp_p_weights['cell.MLPp.5.weight'])
    mlp.cell.MLPp[5].bias.copy_(mlp_p_weights['cell.MLPp.5.bias'])

# freeze MLPp
for param in mlp.cell.MLPp.parameters():
    param.requires_grad = False


optimizer = optim.Adam(mlp.parameters(), lr=5e-3)
criterion = nn.MSELoss().to(DEVICE)
param_count = sum(p.numel() for p in mlp.parameters() if p.requires_grad)
print("Total number of trainable parameters are ",param_count)
# Prepare data
#X = np.linspace(0.0, 1.0, 100).reshape(-1, 1).astype(np.float32)
#Y = np.hstack([np.linspace(0.85, -0.2, 90), np.linspace(-0.25, -0.8, 10)]).reshape(-1, 1).astype(np.float32)
X = inputs_shiffed[train_idx,:,:]
Y = target_shiffed[train_idx,:, np.newaxis]
# Convert data to PyTorch tensors
X_tensor = torch.from_numpy(X).to(DEVICE)
Y_tensor = torch.from_numpy(Y).to(DEVICE)

# Create PyTorch Dataset and DataLoader
dataset = TensorDataset(X_tensor, Y_tensor)
data_loader = DataLoader(dataset, batch_size=30, shuffle=True)
print("I am loading ",len(data_loader))
# Learning rate scheduler
scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: 1 if epoch < 800 else (0.5 if epoch < 1100 else (0.25 if epoch < 2200 else 0.125)))
# scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: 2e-2 if epoch < 800 else (1e-2 if epoch < 1100 else (5e-3 if epoch < 2200 else 1e-3)))
untrained_parameter_value = [mlp.cell.Ro.data.item(), mlp.cell.qMax.data.item()]
print("UnTrained Parameter Value:", untrained_parameter_value)
# Training loop
start = time.time()
num_epochs = NUM_EPOCHS
loss_warm_start = []
for epoch in range(num_epochs):
    mlp.train()
    total_loss = 0.0
    #print("Epochs are ",epoch)
    for inputs, targets in data_loader:
        inputs.to(DEVICE)
        targets.to(DEVICE)

        optimizer.zero_grad()
        # Forward pass
        outputs = mlp(inputs)

        # Compute loss
        loss = criterion(outputs, targets)

        # Backpropagation
        loss.backward()
        torch.nn.utils.clip_grad_norm_(mlp.parameters(), 1.0) #TODO: This sets gradient clipping
        optimizer.step()
        #scheduler.step()
        total_loss += loss.item()

    # Adjust learning rate using scheduler
    #scheduler.step()   <-- no idea why they do this we have ADAM
    # Print epoch statistics
    if epoch % 100 == 0:
        loss_warm_start.append(total_loss / len(data_loader))
        print(f"Epoch {epoch}, Loss: {total_loss / len(data_loader)}, Time : {time.time()-start}, Ro: {mlp.cell.Ro.data.item()}, qMax: {mlp.cell.qMax.data.item()}")
        for param_group in optimizer.param_groups:
            current_learning_rate = param_group['lr']
            print("Current Learning Rate:", current_learning_rate)

# Save model weights
torch.save(mlp.state_dict(), 'torch_train/Ro_qmax_trained_weights.pth')
trained_parameter_value = [mlp.cell.Ro.data.item(), mlp.cell.qMax.data.item()]
print("Trained Parameter Value:", trained_parameter_value)
###### FOR REFERENCE : TRAINING STARTS HERE #########