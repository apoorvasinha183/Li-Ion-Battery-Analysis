import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from battery_data import getDischargeMultipleBatteries
#from BatteryRNNCell_mlp import BatteryRNN
from model import get_model
###### FOR REFERENCE : DATA INGESTION STARTS HERE ##########
def get_data_tensor(data_dict, max_idx_to_use, max_size):
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
    inputs_array = np.array(inputs)
    target_array = np.array(target)

   
    #inputs_tensor = torch.tensor(inputs_array, dtype=torch.float32)
    #target_tensor = torch.tensor(target_array, dtype=torch.float32)

    return inputs_array, target_array

# Load battery data
data_RW = getDischargeMultipleBatteries()
max_idx_to_use = 3 # We are training the battery with constamt current data
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
        target_shiffed[row[0]][time_window_size-row[1]:] = target_array[row[0]][:row[1]]
#dataset = TensorDataset(inputs_tensor.unsqueeze(-1), target_tensor.unsqueeze(-1))
#train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
val_idx = np.linspace(0,35,6,dtype=int)
train_idx = [i for i in np.arange(0,36) if i not in val_idx]

###### FOR REFERENCE : DATA INGESTION ENDS HERE ##########
        
###### FOR REFERENCE : TRAINING STARTS HERE #########
        
# Create the MLP model, optimizer, and criterion
mlp = get_model(batch_input_shape=(inputs_array[train_idx,:,:].shape[0],time_window_size,inputs_array.shape[2]), dt=dt, mlp=True, share_q_r=False, stateful=True)
optimizer = optim.Adam(mlp.parameters(), lr=2e-2)
criterion = nn.MSELoss()

# Prepare data
#X = np.linspace(0.0, 1.0, 100).reshape(-1, 1).astype(np.float32)
#Y = np.hstack([np.linspace(0.85, -0.2, 90), np.linspace(-0.25, -0.8, 10)]).reshape(-1, 1).astype(np.float32)
X = inputs_shiffed[train_idx,:,:]
Y = target_shiffed[train_idx,:,np.newaxis]
# Convert data to PyTorch tensors
X_tensor = torch.from_numpy(X)
Y_tensor = torch.from_numpy(Y)

# Create PyTorch Dataset and DataLoader
dataset = TensorDataset(X_tensor, Y_tensor)
data_loader = DataLoader(dataset, batch_size=1, shuffle=True)

# Learning rate scheduler
scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: 2e-2 if epoch < 800 else (1e-2 if epoch < 1100 else (5e-3 if epoch < 2200 else 1e-3)))

# Training loop
num_epochs = 5000
for epoch in range(num_epochs):
    mlp.train()
    total_loss = 0.0

    for inputs, targets in data_loader:
        optimizer.zero_grad()

        # Forward pass
        outputs = mlp(inputs)

        # Compute loss
        loss = criterion(outputs, targets)

        # Backpropagation
        loss.backward()
        optimizer.step()
        #scheduler.step()
        total_loss += loss.item()

    # Adjust learning rate using scheduler
    scheduler.step()

    # Print epoch statistics
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {total_loss / len(data_loader)}")

# Save model weights
torch.save(mlp.state_dict(), 'torch_train/mlp_trained_weights.pth')

# Plot predictions
mlp.eval()
with torch.no_grad():
    pred = mlp(X_tensor).numpy()

plt.plot(X, Y, color='gray')
plt.plot(X, pred)
plt.grid()
plt.show()






###### FOR REFERENCE : TRAINING ENDS HERE #########        