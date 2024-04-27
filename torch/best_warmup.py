# Does it matter which curve I choose?
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
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CHECK = 2
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.MLPp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1, 8),
            nn.Tanh(),
            nn.Linear(8, 4),
            nn.Tanh(),
            nn.Linear(4, 1)
        )
    def forward(self, x):
        x = self.MLPp(x)
        #x = self.activation(self.fc1(x))
        #x = self.activation(self.fc2(x))
        #x = self.fc3(x)
        return x
# Method which trains based on the knee location
def sweet_warmup_spot(knee):
    # knee is the inflexion point 10 - 90
    mlp = MLP().to(DEVICE)
    optimizer = optim.Adam(mlp.parameters(), lr=2e-2)
    criterion = nn.MSELoss()

    # Prepare data
    X = np.linspace(0.0, 1.0, 100).reshape(-1, 1).astype(np.float32)
    Y = np.hstack([np.linspace(0.85, -0.2, knee), np.linspace(-0.25, -0.8, 100-knee)]).reshape(-1, 1).astype(np.float32)

    # Convert data to PyTorch tensors
    X_tensor = torch.from_numpy(X).to(DEVICE)
    Y_tensor = torch.from_numpy(Y).to(DEVICE)

    # Create PyTorch Dataset and DataLoader
    dataset = TensorDataset(X_tensor, Y_tensor)
    data_loader = DataLoader(dataset, batch_size=1, shuffle=True)

    # Learning rate scheduler
    #scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: 2e-2 if epoch < 800 else (1e-2 if epoch < 1100 else (5e-3 if epoch < 2200 else 1e-3)))
    print("Shapes are ",X_tensor.size(),Y_tensor.size())
    # Training loop
    num_epochs = 10001
    for epoch0 in range(num_epochs):
        mlp.train()
        total_loss = 0.0
        #print("Epochs are ",epoch)
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
        #scheduler.step()

        # Print epoch statistics
        if epoch0 % 100 == 0:
            print(f"Epoch {epoch0}, Loss: {total_loss / len(data_loader)}")

    # Save model weights
            
    PATH = 'torch_train/mlp_initial_weights.pth'
    torch.save({
        'epoch': epoch0,
        'model_state_dict': mlp.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
        }, PATH)
    return    
# Actual Training
def train(seed):
    # Return the model
    NUM_EPOCHS = 2001
    #NUM_CHECK = 1 # Between 1 and 6 .How many batteries do you want to evaluate
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
            #print("shifted targets are ",target_shiffed[row[0]])
            target_shiffed[row[0]][time_window_size-row[1]:] = target_array[row[0]][:row[1]]
    #dataset = TensorDataset(inputs_tensor.unsqueeze(-1), target_tensor.unsqueeze(-1))
    #train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_idx = np.linspace(0,35,6,dtype=int)
    #val_idx = np.array([0])
    train_idx = [i for i in np.arange(0,36) if i not in val_idx]
    #train_idx = np.array([1])
    ###### FOR REFERENCE : DATA INGESTION ENDS HERE ##########
            
    ###### FOR REFERENCE : TRAINING STARTS HERE #########
            
    # Create the MLP model, optimizer, and criterion
    mlp = get_model(dt=dt, mlp=True, share_q_r=False, stateful=True).to(DEVICE)
    optimizer = optim.Adam(mlp.parameters(), lr=2e-2)
    criterion = nn.MSELoss().to(DEVICE)
    param_count = sum(p.numel() for p in mlp.parameters() if p.requires_grad)
    print("Total number of trainable parameters are ",param_count)
    # Prepare data
    #X = np.linspace(0.0, 1.0, 100).reshape(-1, 1).astype(np.float32)
    #Y = np.hstack([np.linspace(0.85, -0.2, 90), np.linspace(-0.25, -0.8, 10)]).reshape(-1, 1).astype(np.float32)
    X = inputs_shiffed[train_idx,:,:]
    Y = target_shiffed[train_idx,:,np.newaxis]
    # Convert data to PyTorch tensors
    X_tensor = torch.from_numpy(X).to(DEVICE)
    Y_tensor = torch.from_numpy(Y).to(DEVICE)

    # Create PyTorch Dataset and DataLoader
    dataset = TensorDataset(X_tensor, Y_tensor)
    data_loader = DataLoader(dataset, batch_size=30, shuffle=True)
    print("I am loading ",len(data_loader))
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: 2e-2 if epoch < 800 else (1e-2 if epoch < 1100 else (5e-3 if epoch < 2200 else 1e-3)))
    untrained_parameter_value = [mlp.cell.MLPp[1].weight.data,mlp.cell.Ro.data.item()]
    print("UnTrained Parameter Value:", untrained_parameter_value)
    # Training loop
    start = time.time()
    num_epochs1 = 1001
    loss_warm_start = []
    for epoch in range(num_epochs1):
        #print("This is epoch number ",epoch)
        mlp.train()
        total_loss = 0.0
        #print("Epochs are ",epoch)
        for inputs, targets in data_loader:
            inputs.to(DEVICE)
            targets.to(DEVICE)
            # if torch.cuda.is_available():
            #     inputs = inputs.cuda()
            #     targets = targets.cuda()

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
        #scheduler.step()   <-- no idea why they do this we have ADAM

        # Print epoch statistics
        if epoch % 100 == 0:
            loss_warm_start.append(total_loss / len(data_loader))
            print(f"Epoch {epoch}, Loss: {total_loss / len(data_loader)}, Time : {time.time()-start}")

        # Save model weights
    fn = 'torch_train/mlp_trained_weights_'+str(seed)+'.pth'
    torch.save(mlp.state_dict(), fn)
        #trained_parameter_value = [mlp.cell.MLPp[1].weight.data,mlp.cell.Ro.data.item()]
        #print("Trained Parameter Value:", trained_parameter_value)
        # Plot predictions
        #mlp.eval()
        #with torch.no_grad():
        #    pred = mlp(X_tensor).cpu().numpy()

        #plt.plot(X, Y, color='gray')
        #plt.plot(X, pred)
        #plt.grid()
        #plt.show()
    mlp.eval()
    # Time for the test set
    X = inputs_array[val_idx,:,:]
    # For confidential reasons
    shape_X = np.shape(X)
    print(shape_X)

    Y = target_array[val_idx,:,np.newaxis]  
    X_tensor = torch.from_numpy(X).to(DEVICE)
    Y_tensor = torch.from_numpy(Y).to(DEVICE)
    with torch.no_grad():
        pred = mlp(X_tensor).cpu().numpy()

    return X_tensor,Y_tensor,pred


# Train with different loops and evaluate
kink_loc = [85,90,95]
#kink_loc = [50,60]
for kinks in kink_loc:
    sweet_warmup_spot(kinks)
    input,output,prediction = train(kinks)
    # Eval
    for i in range(NUM_CHECK):
        plt.plot(prediction[i,:,0],linestyle='dashed',label =str(kinks)+"Battery "+str(i))
        plt.plot(output[i,:,0])
plt.ylabel('Voltage(V)')
plt.grid()
plt.legend()

plt.xlabel('Time')
plt.savefig('figures/predictionvsrealitydifferentwarmups.png')
plt.show()