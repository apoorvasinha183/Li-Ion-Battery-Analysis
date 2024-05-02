import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
from battery_data import BatteryDataFile, getDischargeMultipleBatteries, DATA_PATH, BATTERY_FILES
import time

#from BatteryRNNCell_mlp import BatteryRNN
from model import get_model
#DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE = torch.device("cpu")
NUM_EPOCHS = 1
#BATTERY = 3

###### FOR REFERENCE : DATA INGESTION STARTS HERE ##########
data_RW = getDischargeMultipleBatteries(varnames=['voltage', 'current', 'time'], discharge_type='discharge (random walk)')

max_size = 0
num_seq = 36


batteries =[1,2,3,4,5,6,7,8,9,10,11,12]
#batteries =[2]

for BATTERY in batteries:
    loss_history_batt = []
    rw_data = data_RW[BATTERY]
    time_ = np.hstack([rw_data[2][i] for i in range(len(rw_data[2]))])
    time_ = time_ - time_[0]
    current_inputs = np.hstack([rw_data[1][i] for i in range(len(rw_data[1]))])
    voltage_target = np.hstack([rw_data[0][i] for i in range(len(rw_data[0]))])
    inputs = []
    inputs_time = []
    target = []
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
    #print("vali_idx is ",val_idx)
    train_idx = [i for i in np.arange(0,36) if i not in val_idx]
    ###### FOR REFERENCE : DATA INGESTION ENDS HERE ##########




    ###### FOR REFERENCE : MODEL LOADING STARTS HERE ##########

    mlp = get_model(dt=dt, mlp=True, share_q_r=False, stateful=True).to(DEVICE)
    weights_path = f'torch_train/mlp_trained_weights_bias_correction_{BATTERY}_complete.pth'
    # weights_path = "torch_train/Ro_qmax_trained_battery_1.pth"
    mlp_p_weights = torch.load(weights_path)

    print(f'Ro: {mlp_p_weights["cell.Ro"]}, qMax: {mlp_p_weights["cell.qMax"]}')

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
    X = inputs[val_idx,:,:]
    Y = target[val_idx,:, np.newaxis]
    X_tensor = torch.from_numpy(X).to(DEVICE)
    Y_tensor = torch.from_numpy(Y).to(DEVICE)
    with torch.no_grad():
        pred = mlp(X_tensor).cpu().numpy()
    criterion = nn.MSELoss()
    
    #plt.plot(X, Y, color='gray')
    #print("Predictions have shape ",pred.shape)
    loss_battery = []
    max_loss_battery = []
    max_loss = 0
    for i in range(X.shape[0]):
        fig = plt.figure(f"{BATTERY}",figsize=(10, 5))
        ax1 = fig.add_subplot(111)
        ax1.set_xlabel('time')
        ax1.set_ylabel('Voltage [V]')
        ax1.set_ylim(0,4.5)
        ax2 = ax1.twinx()
        ax1.plot(pred[i,:,0], linestyle='dashed', color='red', label='Voltage Prediction')
        ax1.plot(Y_tensor[i,:,0], color='blue', label='Voltage Measured')
        ax1.legend()
        ax2.plot(X_tensor[i,:,0], color='purple', label='Current Measured', alpha=0.5)
        #loss = F.mse_loss(torch.tensor(pred[i,:,0]),Y_tensor[i,:,0])
        prediction_tensor = torch.tensor(pred[i,:,0])
        target_tensor = Y_tensor[i,:,0]
        # Create masks to identify non-NaN values
        mask_prediction = ~torch.isnan(prediction_tensor)
        mask_target = ~torch.isnan(target_tensor)

        # Combine masks to identify elements that are not NaN in both prediction and target
        mask_valid = mask_prediction & mask_target

        # Filter tensors to exclude NaN values
        prediction_valid = prediction_tensor[mask_valid]
        target_valid = target_tensor[mask_valid]
        loss = F.mse_loss(prediction_valid, target_valid)
        if loss.item() > max_loss:
            max_loss = loss.item()
        max_loss_battery.append(max_loss)    
        #print("MSE loss is ",loss.item())
        loss_battery.append(loss.item())
        #print("Prediction tensor has size ",pred[i,:,0].shape)
        #print("Target has size ",Y_tensor[i,:,0].shape)
        ax2.legend()
        ax2.set_ylabel('Current [I]')
        ax2.yaxis.label.set_color('purple')
        ax2.spines["right"].set_edgecolor('purple')
        ax2.tick_params(axis='y', colors='purple')

        plt.savefig(f'figures/apoorva_baterry_{BATTERY}_unshiffed_{i}_final_project_submission_50samples.png')
        plt.show()
        #plt.close()
    loss_battery = np.array(loss_battery)
    max_loss_battery = np.array(max_loss_battery)
    print(f"Battery {BATTERY} has an avergae MSE Test Loss of {np.mean(loss_battery)} and maximum MSE error of {np.max(loss_battery)}")    
    #plt.plot(1000*max_loss_battery,label = f'Battery {BATTERY}')
#plt.ylabel("MSE Loss(in mVolts)")
#plt.xlabel("Number of cycles ")
#plt.title("Cumulative max MSE loss in Random Walks based on only reference discharge trained models")
#plt.legend(loc="upper right")
#plt.savefig(f'figures/apoorva_baterry_fullmax_loss_history_on_dc_data.png')
#plt.show()    
    ######## Validation ends here ##########

