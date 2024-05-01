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
NUM_EPOCHS = 2000
BATTERY = 1

###### FOR REFERENCE : DATA INGESTION STARTS HERE ##########
data_RW = getDischargeMultipleBatteries(varnames=['voltage', 'current', 'time'], discharge_type='discharge (random walk)')

max_size = 0
num_seq = 36

inputs = []
inputs_time = []
target = []


rw_data = data_RW[BATTERY]
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

val_idx = np.linspace(0,35,6,dtype=int)
train_idx = [i for i in np.arange(0,36) if i not in val_idx]
###### FOR REFERENCE : DATA INGESTION ENDS HERE ##########





###### FOR REFERENCE : TRAINING STARTS HERE #########
        
# Create the MLP model, optimizer, and criterion
mlp = get_model(dt=dt, mlp_trainable=False, share_q_r=False, stateful=True).to(DEVICE)

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
X = inputs_shiffed[train_idx,:,:]
Y = target_shiffed[train_idx,:, np.newaxis]
# Convert data to PyTorch tensors
X_tensor = torch.from_numpy(X).to(DEVICE)
Y_tensor = torch.from_numpy(Y).to(DEVICE)



X_test = inputs_shiffed[val_idx,:,:]
Y_test = target_shiffed[val_idx,:,np.newaxis]
# Convert data to PyTorch tensors
X_test_tensor = torch.from_numpy(X_test).to(DEVICE)
Y_test_tensor = torch.from_numpy(Y_test).to(DEVICE)


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
loss_val_warm_start = []

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
        # torch.nn.utils.clip_grad_norm_(mlp.parameters(), 1.0) #TODO: This sets gradient clipping
        optimizer.step()
        #scheduler.step()
        total_loss += loss.item()

    # Adjust learning rate using scheduler
    #scheduler.step()   <-- no idea why they do this we have ADAM
    # Print epoch statistics
    if epoch % 100 == 0:
        loss_warm_start.append(total_loss / len(data_loader))

        # evaluate accuracy at end of training
        with torch.no_grad():
            mlp.eval()
            Y_pred = mlp(X_test_tensor)
            loss_val = criterion(Y_pred, Y_test_tensor)
            loss_val_warm_start.append(loss_val.item())

        print(f"Epoch {epoch}, train Loss: {total_loss / len(data_loader)}, val Loss: {loss_val}, Time : {time.time()-start}, Ro: {mlp.cell.Ro.data.item()}, qMax: {mlp.cell.qMax.data.item()}")
        start = time.time()

# Save model weights
torch.save(mlp.state_dict(), 'torch_train/Ro_qmax_trained_weights_battery_{BATTERY}.pth')
trained_parameter_value = [mlp.cell.Ro.data.item(), mlp.cell.qMax.data.item()]
print("Trained Parameter Value:", trained_parameter_value)
###### FOR REFERENCE : TRAINING STARTS HERE #########

plt.figure(figsize=(4, 2))
plt.plot(loss_warm_start, label='train Loss')
plt.plot(loss_val_warm_start, label='val loss')
plt.legend()
plt.ylabel('MSE(Loss)')
plt.xlabel('Epoch(unit of 100)')
# plt.grid()
plt.savefig('figures/loss_battery_1_random_train.png')
plt.show()

# ######## Validation is done here ##########
# mlp.eval()
# # Time for the test set
# X = inputs[val_idx,:,:]
# Y = target[val_idx,:, np.newaxis]
# X_tensor = torch.from_numpy(X).to(DEVICE)
# Y_tensor = torch.from_numpy(Y).to(DEVICE)
# with torch.no_grad():
#     pred = mlp(X_tensor).cpu().numpy()

# #plt.plot(X, Y, color='gray')
# print("Predictions have shape ",pred.shape)
# for i in range(X.shape[0]):
#     fig = plt.figure(figsize=(10, 5))
#     ax1 = fig.add_subplot(111)
#     ax1.set_xlabel('time')
#     ax1.set_ylabel('Voltage [V]')
#     ax2 = ax1.twinx()
#     ax1.plot(pred[i,:,0], linestyle='dashed', color='red', label='Voltage Prediction')
#     ax1.plot(Y_tensor[i,:,0], color='blue', label='Voltage Measured')
#     ax1.legend()
#     ax2.plot(X_tensor[i,:,0], color='purple', label='Current Measured', alpha=0.5)
#     ax2.legend()
#     ax2.set_ylabel('Current [I]')
#     ax2.yaxis.label.set_color('purple')
#     ax2.spines["right"].set_edgecolor('purple')
#     ax2.tick_params(axis='y', colors='purple')

#     plt.savefig(f'figures/predictionvsreality_random_walk_unshiffed{i}_Ro_qMax_battery_{BATTERY}.png')
#     # plt.show()
# ######## Validation is done here ##########


# ######## Validation is done here ##########
# mlp.eval()
# # Time for the test set
# X = inputs_shiffed[val_idx,:,:]
# Y = target_shiffed[val_idx,:,np.newaxis]
# X_tensor = torch.from_numpy(X).to(DEVICE)
# Y_tensor = torch.from_numpy(Y).to(DEVICE)
# with torch.no_grad():
#     pred = mlp(X_tensor).cpu().numpy()

# #plt.plot(X, Y, color='gray')
# print("Predictions have shape ",pred.shape)
# for i in range(X.shape[0]):
#     fig = plt.figure(figsize=(10, 5))
#     ax1 = fig.add_subplot(111)
#     ax1.set_xlabel('time')
#     ax1.set_ylabel('Voltage [V]')
#     ax2 = ax1.twinx()
#     ax1.plot(pred[i,:,0], linestyle='dashed', color='red', label='Voltage Prediction')
#     ax1.plot(Y_tensor[i,:,0], color='blue', label='Voltage Measured')
#     ax1.legend()
#     ax2.plot(X_tensor[i,:,0], color='purple', label='Current Measured', alpha=0.5)
#     ax2.legend()
#     ax2.set_ylabel('Current [I]')
#     ax2.yaxis.label.set_color('purple')
#     ax2.spines["right"].set_edgecolor('purple')
#     ax2.tick_params(axis='y', colors='purple')

#     plt.savefig(f'figures/predictionvsreality_random_walk_shiffed{i}_Ro_qMax_battery_{BATTERY}.png')
#     # plt.show()
# ######## Validation is done here ##########