import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from battery_data import getDischargeMultipleBatteries
from model import get_model


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NUM_CHECK = 1


# Data function
print("Loading data...")

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

    return inputs_array, target_array

# Load battery data
data_RW = getDischargeMultipleBatteries()
max_idx_to_use = 5 # Number of training batteries to use

max_size = np.max([v[0, 0].shape[0] for k, v in data_RW.items()])

# Discretization used
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
val_idx = np.linspace(0,max_idx_to_use*12-1,6,dtype=int)
train_idx = [i for i in np.arange(0,max_idx_to_use*12) if i not in val_idx]

        
###### FOR REFERENCE : TRAINING STARTS HERE #########
        
# Create the MLP model, optimizer, and criterion
mlp = get_model(dt=dt, mlp=True, share_q_r=False, stateful=True, D_trainable=False, version="naive_PINN").to(DEVICE)
optimizer = optim.Adam(mlp.parameters(), lr=1e-3) #weight_decay=1e-5
criterion = nn.MSELoss()
l2_lambda = 0

# Prepare data
X = inputs_shiffed[train_idx,:,:]
Y = target_shiffed[train_idx,:,np.newaxis]


# Convert data to PyTorch tensors
X_tensor = torch.from_numpy(X).to(DEVICE).float()
Y_tensor = torch.from_numpy(Y).to(DEVICE).float()

print("Loaded data and preprocessed!")

# Create PyTorch Dataset and DataLoader
dataset = TensorDataset(X_tensor, Y_tensor)
data_loader = DataLoader(dataset, batch_size=15, shuffle=True)

# Learning rate scheduler
scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: 1 if epoch < 50 else (0.2 if epoch < 150 else (0.05 if epoch < 1200 else 0.025)))

# Training loop
print("Starting training...")
num_epochs = 200
for epoch in range(num_epochs):

    mlp.train()

    total_loss = 0.0

    for inputs, targets in data_loader:
        inputs.to(DEVICE)
        targets.to(DEVICE)
        optimizer.zero_grad()
        
       
        # Forward pass
        pred = mlp(inputs)
        
        pred = pred.permute([0,2,1])

        # Compute loss
        loss = criterion(pred, targets)

        #l2_reg = sum(torch.norm(param)**2 for param in mlp.parameters())

        # Total loss with L2 regularization

        # Backpropagation
        loss.backward()

        """for name, param in mlp.named_parameters():
            if param.grad is not None:
                print(f"Gradient of {name}: {param.grad}")"""

        # Clip Gradients
        #torch.nn.utils.clip_grad_norm_(mlp.parameters(), 1.0)
        
        optimizer.step()
        total_loss += loss.item()
        

    # Adjust learning rate using scheduler
    scheduler.step()

    # Print epoch statistics
    if epoch % 1 == 0:
        print(f"Epoch {epoch}, Total Loss: {total_loss / len(data_loader)}")  #


# Save model weights
torch.save(mlp.state_dict(), 'torch_train/trained_weights_PINN.pth')

mlp.eval()
# Time for the test set
X = inputs_array[val_idx,:,:]
Y = target_array[val_idx,:,np.newaxis]  
X_tensor = torch.from_numpy(X).to(DEVICE)
Y_tensor = torch.from_numpy(Y).to(DEVICE)
with torch.no_grad():
    pred = mlp(X_tensor).cpu().numpy()
print(pred[1,:])
print(Y_tensor[1,:,0])

for i in range(NUM_CHECK):
    plt.plot(pred[i,:],linestyle='dashed')
    #plt.plot(Y_tensor[i,:,0])
plt.ylabel('Voltage(V)')
plt.grid()
#plt.ylim(2, 5)
plt.xlabel('Time')
plt.savefig('figures/predictionvsreality.png')
plt.show()


###### FOR REFERENCE : TRAINING ENDS HERE #########        