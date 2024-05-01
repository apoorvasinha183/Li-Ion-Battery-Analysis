import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from battery_data import getDischargeMultipleBatteries
import time

#from BatteryRNNCell_mlp import BatteryRNN
from model import get_model
#DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE = torch.device("cpu")
EXPERIMENT = False #Compares and plots watm-start vs random initialization
NUM_EPOCHS = 1001
NUM_CHECK = 1 # Between 1 and 6 .How many batteries do you want to evaluate


###### FOR REFERENCE : DATA INGESTION STARTS HERE ##########
def shift_data(data_dict, max_idx_to_use, max_size):
    inputs = None
    target = None

    for k, v in data_dict.items():
        for i, d in enumerate(v[1, :max_idx_to_use]):
            prep_inp = np.full(max_size, np.nan)
            prep_target = np.full(max_size, np.nan)
            prep_inp[:len(d)] = d   # Current Sequence
            prep_target[:len(v[0, :][i])] = v[0, :][i]  # Voltage sequence
            if inputs is None:
                inputs = prep_inp
                target = prep_target
            else:
                inputs = np.vstack([inputs, prep_inp])   
                target = np.vstack([target, prep_target])

    inputs = inputs[:,:,np.newaxis]
    time_window_size = inputs.shape[1]
    num_curves = inputs.shape[0]

    # move timesteps with earlier EOD  --- This move aligns the sequence lengths of the discharge sequences
    # Batteries which discharge earlier are shifted such that all discharges happen simultaneously.
    # Assume these batteries are drawing 0 current till that time .
    EOD = 3.2
    V_0 = 4.19135029  # V adding when I=0 for the shift
    inputs_shifted = inputs.copy()
    inputs_us = inputs_shifted.copy()
    target_shifted = target.copy()
    targets_us = target_shifted.copy()
    reach_EOD = np.ones(num_curves, dtype=int) * time_window_size
    for row in np.argwhere((target<EOD) | (np.isnan(target))):
        if reach_EOD[row[0]]>row[1]:
            reach_EOD[row[0]]=row[1]
            inputs_shifted[row[0],:,0] = np.zeros(time_window_size) # Put zeros to keep data sane
            inputs_shifted[row[0],:,0][time_window_size-row[1]:] = inputs[row[0],:,0][:row[1]]
            target_shifted[row[0]] = np.ones(time_window_size) * target[row[0]][0]
            #print("shifted targets are ",target_shifted[row[0]])
            target_shifted[row[0]][time_window_size-row[1]:] = target[row[0]][:row[1]]
   
    return inputs_shifted, target_shifted,inputs_us,targets_us


# Load battery data
data_RW = getDischargeMultipleBatteries(discharge_type='reference discharge')
max_idx_to_use = 3 # We are training the battery with constamt current data
max_size = np.max([v[0, 0].shape[0] for k, v in data_RW.items()])
dt = np.diff(data_RW[1][2, 0])[1]

# Get data tensors
inputs_shifted, target_shifted,inputs_us,targets_us = shift_data(data_RW, max_idx_to_use, max_size)
val_idx = np.linspace(0,35,6,dtype=int)
train_idx = [i for i in np.arange(0,36) if i not in val_idx]
###### FOR REFERENCE : DATA INGESTION ENDS HERE ##########



###### FOR REFERENCE : MODEL LOADING STARTS HERE ##########
mlp = get_model(dt=dt, mlp=True, share_q_r=False, stateful=True).to(DEVICE)
weights_path = 'torch_train/mlp_trained_weights_bias_correction_4.pth'
mlp_p_weights = torch.load(weights_path)

with torch.no_grad():
    mlp.cell.MLPp[1].weight.copy_(mlp_p_weights["cell.MLPp.1.weight"])
    mlp.cell.MLPp[1].bias.copy_(mlp_p_weights['cell.MLPp.1.bias'])
    mlp.cell.MLPp[3].weight.copy_(mlp_p_weights['cell.MLPp.3.weight'])
    mlp.cell.MLPp[3].bias.copy_(mlp_p_weights['cell.MLPp.3.bias'])
    mlp.cell.MLPp[5].weight.copy_(mlp_p_weights['cell.MLPp.5.weight'])
    mlp.cell.MLPp[5].bias.copy_(mlp_p_weights['cell.MLPp.5.bias'])

    
#Ro_qmax_path ='torch_train/Ro_qmax_trained_weights.pth'
Ro_qmax_path = 'torch_train/mlp_trained_weights_bias_correction_4.pth'
Ro_qmax = torch.load(weights_path)

with torch.no_grad():
    mlp.cell.qMax.copy_(Ro_qmax['cell.qMax']) 
    mlp.cell.Ro.copy_(Ro_qmax['cell.Ro']) 

###### FOR REFERENCE : MODEL LOADING ENDS HERE ##########


######## Validation is done here ##########
mlp.eval()
# Time for the test set
X = inputs_us[val_idx,:,:]
Y = targets_us[val_idx,:,np.newaxis]  
X_tensor = torch.from_numpy(X).to(DEVICE)
Y_tensor = torch.from_numpy(Y).to(DEVICE)
with torch.no_grad():
    pred = mlp(X_tensor).cpu().numpy()

#plt.plot(X, Y, color='gray')
print("Predictions have shape ",pred.shape)
for i in range(NUM_CHECK):
    fig = plt.figure(figsize=(10, 5))
    ax1 = fig.add_subplot(111)
    ax1.set_xlabel('time')
    ax1.set_ylabel('Voltage [V]')
    ax2 = ax1.twinx()
    #ax1.plot(pred[i,:,0], linestyle='dashed', color='red', label='Voltage Prediction')
    ax1.plot(Y_tensor[i,:,0], color='blue', label='Voltage Measured')
    ax2.plot(X_tensor[i,:,0], color='purple', label='Current Measured', alpha=0.5)
    ax2.set_ylabel('Current [I]')
    ax2.yaxis.label.set_color('purple')
    ax2.spines["right"].set_edgecolor('purple')
    ax2.tick_params(axis='y', colors='purple')
    ax2.set_ylim(0,1.1)
    #plt.savefig(f'figures/predictionvsreality_reference_discharge_{i}.png')
    plt.title("One Complete Discharge on Constant Load")
    plt.savefig(f'example_cycle.png')
    plt.show()
######## Validation is done here ##########