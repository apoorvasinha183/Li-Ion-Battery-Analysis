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
DEVICE =torch.device("cpu")
EXPERIMENT = False #Compares and plots watm-start vs random initialization
NUM_EPOCHS = 1001
NUM_CHECK = 1 # Between 1 and 6 .How many batteries do you want to evaluate
BATTERY = 1
Validate = True

###### FOR REFERENCE : DATA INGESTION STARTS HERE ##########
def get_data_tensor(data_dict, max_idx_to_use, max_size):
    inputs = None
    target = None

    k = BATTERY  
    v = data_dict[k]
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

    inputs_array = np.array(inputs)
    target_array = np.array(target)

    #inputs_tensor = torch.tensor(inputs_array, dtype=torch.float32)
    #target_tensor = torch.tensor(target_array, dtype=torch.float32)

    return inputs_array, target_array

# Load battery data
data_RW = getDischargeMultipleBatteries()
max_idx_to_use = 36 # We are training the battery with constamt current data #This is to force nly one battery data
max_size = np.max([v[0, 0].shape[0] for k, v in data_RW.items()])
dt = np.diff(data_RW[1][2, 0])[1]
# Get data tensors
inputs_array, target_array = get_data_tensor(data_RW, max_idx_to_use, max_size)

inputs_array = inputs_array[:,:,np.newaxis]
time_window_size = inputs_array.shape[1]
BATCH_SIZE = inputs_array.shape[0]

# move timesteps with earlier EOD  --- This move aligns the sequence lengths of the discharge sequences
# Batteries which discharge earlier are shifted such that all discharges happen simultaneously.
# Assume these batteries are drawing 0 current till that time .
EOD = 3.2
V_0 = 4.19135029  # V adding when I=0 for the shift
inputs_shiffed = inputs_array.copy()
target_shiffed = target_array.copy()
reach_EOD = np.ones(BATCH_SIZE, dtype=int) * time_window_size
for row in np.argwhere((target_array<EOD) | (np.isnan(target_array))):
    if reach_EOD[row[0]]>row[1]:
        reach_EOD[row[0]]=row[1]
        inputs_shiffed[row[0],:,0] = np.zeros(time_window_size) # Put zeros to keep data sane
        inputs_shiffed[row[0],:,0][time_window_size-row[1]:] = inputs_array[row[0],:,0][:row[1]]
        target_shiffed[row[0]] = np.ones(time_window_size) * target_array[row[0]][0]
        #print("shifted targets are ",target_shiffed[row[0]])
        target_shiffed[row[0]][time_window_size-row[1]:] = target_array[row[0]][:row[1]]
#dataset = TensorDataset(inputs_tensor.unsqueeze(-1), target_tensor.unsqueeze(-1))
#train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
val_idx = np.linspace(0,35,6,dtype=int)
#val_idx = np.array([0])
train_idx = [i for i in np.arange(0,36) if i not in val_idx]
#train_idx = [0]
#train_idx = np.array([1])
###### FOR REFERENCE : DATA INGESTION ENDS HERE ##########
        





###### FOR REFERENCE : MODEL loading starts here ##########
# Create the MLP model, optimizer, and criterion
mlp = get_model(dt=dt, mlp_trainable=False, share_q_r=False, stateful=True).to(DEVICE)

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


###### FOR REFERENCE : MODEL loading ends here ##########






    
######## Validation is done here ##########
if Validate:
    mlp.eval()
    # Time for the test set
    X = inputs_shiffed[val_idx,:,:]
    # For confidential reasons
    shape_X = np.shape(X)
    # print(shape_X)

    Y = target_shiffed[val_idx,:,np.newaxis]  
    X_tensor = torch.from_numpy(X).to(DEVICE)
    Y_tensor = torch.from_numpy(Y).to(DEVICE)
    with torch.no_grad():
        pred = mlp(X_tensor).cpu().numpy()

    #plt.plot(X, Y, color='gray')
    # print("Predictions have shape ",pred.shape)
    plt.figure(figsize=(4, 2))
    for i in range(NUM_CHECK):
        plt.plot(pred[i,:,0],linestyle='dashed')
        plt.plot(Y_tensor[i,:,0])
    plt.ylabel('Voltage(V)')
    plt.grid()

    plt.xlabel('Time')
    plt.savefig('figures/predictionvsreality.png')
    plt.show()

######## Validation is done here ##########
