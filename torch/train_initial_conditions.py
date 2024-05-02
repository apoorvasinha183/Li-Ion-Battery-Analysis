import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from battery_data import getDischargeMultipleBatteries
import sys
import time
#from BatteryRNNCell_mlp import BatteryRNN
from model import get_model
#DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE =torch.device("cpu")
EXPERIMENT = False #Compares and plots watm-start vs random initialization
NUM_EPOCHS = 1001
NUM_CHECK = 1 # Between 1 and 6 .How many batteries do you want to evaluate
battery_num = [1,2,3,4,5,6,7,8]
#battery_num=[2]
Validate = False
Validate_one = True
Sanity = True
###### FOR REFERENCE : DATA INGESTION STARTS HERE ##########
def get_data_tensor(data_dict, max_idx_to_use, max_size):
    inputs = None
    target = None
    dummy = []
    for k, v in data_dict.items():
        
        print("k is ",k) # Battery Number track
        for i, d in enumerate(v[1, :max_idx_to_use]):
            prep_inp = np.full(max_size, np.nan)
            prep_target = np.full(max_size, np.nan)
            prep_inp[:len(d)] = d   # Current Sequence
            prep_target[:len(v[0, :][i])] = v[0, :][i]  # Voltage sequence
            dummy.append(k)
            if inputs is None:
                inputs = prep_inp
                target = prep_target
            else:
                inputs = np.vstack([inputs, prep_inp])   
                target = np.vstack([target, prep_target])
               
    inputs_array = np.array(inputs)
    target_array = np.array(target)

    #print("I see the batteries as ",dummy)
    #inputs_tensor = torch.tensor(inputs_array, dtype=torch.float32)
    #target_tensor = torch.tensor(target_array, dtype=torch.float32)

    return inputs_array, target_array

# Load battery data
data_RW = getDischargeMultipleBatteries()
max_idx_to_use = 60 # We are training the battery with constamt current data #This is to force nly one battery data
max_size = np.max([v[0, 0].shape[0] for k, v in data_RW.items()])
#max_size = 900 #i want to teach the models a step response
dt = np.diff(data_RW[1][2, 0])[1]
# Get data tensors
inputs_array, target_array = get_data_tensor(data_RW, max_idx_to_use, max_size)

inputs_array = inputs_array[:,:,np.newaxis]
time_window_size = inputs_array.shape[1]
print("Window ssize is ",time_window_size)
#time_window_size = 1200
BATCH_SIZE = inputs_array.shape[0]
#sys.exit()
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
if not Sanity:
    for battery in battery_num:
        print("Fixing bias for Battery Number ",battery)
        start = (battery-1)*max_idx_to_use
        end = battery*max_idx_to_use
        val_idx = np.linspace(start,end-1,6,dtype=int)
        #val_idx = np.array([0])
        train_idx = [i for i in np.arange(start,end) if i not in val_idx]
        #train_idx = [0]
        #train_idx = np.array([1])
        ###### FOR REFERENCE : DATA INGESTION ENDS HERE ##########
                
        ###### FOR REFERENCE : TRAINING STARTS HERE #########
                
        # Create the MLP model, optimizer, and criterion
        mlp = get_model(dt=dt, mlp=True, share_q_r=False, stateful=True,mlp_trainable=False).to(DEVICE)
        #mlp.train()
        optimizer = optim.Adam(mlp.parameters(), lr=5e-3)
        criterion = nn.MSELoss().to(DEVICE)
        loss_fn = lambda y_true, y_pred: torch.max(torch.abs(y_true - y_pred)) # To minimize the maximum voltage error
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
        print("Xtesor has shape ",X_tensor.shape)
        # Create PyTorch Dataset and DataLoader
        dataset = TensorDataset(X_tensor, Y_tensor)
        data_loader = DataLoader(dataset, batch_size=54, shuffle=True) #TODO : Make it 30 again
        print("I am loading ",len(data_loader))
        # Learning rate scheduler
        #scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: 1 if epoch < 800 else (0.5 if epoch < 1100 else (0.25 if epoch < 2200 else 0.125)))
        untrained_parameter_value = [mlp.cell.qMax.data.item(),mlp.cell.Ro.data.item()]
        print("UnTrained Parameter Value:", untrained_parameter_value)
        # Training loop
        start = time.time()
        num_epochs = NUM_EPOCHS
        loss_warm_start = []
        #optimizer = optim.Adam(mlp.parameters(), lr=0.005)
        for epoch in range(num_epochs):
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
                #loss = loss_fn(outputs,targets) #Custom loss
                # Backpropagation
                loss.backward()
                #torch.nn.utils.clip_grad_norm_(mlp.parameters(), 1.0) #TODO: This sets gradient clipping
                optimizer.step()
                
                #scheduler.step()
                total_loss += loss.item()

            # Adjust learning rate using scheduler
            #scheduler.step()   

            # Print epoch statistics
            if epoch % 100 == 0:
                loss_warm_start.append(total_loss / len(data_loader))
                print(f"Epoch {epoch}, Loss: {total_loss / len(data_loader)}, Time : {time.time()-start}")
                for param_group in optimizer.param_groups:
                    current_learning_rate = param_group['lr']
                    print("Current Learning Rate:", current_learning_rate)
                trained_parameter_value = [mlp.cell.qMax.data.item(),mlp.cell.Ro.data.item()]
                print("Trained Parameter Value right now :", trained_parameter_value)    

        # Save model weights
        torch.save(mlp.state_dict(), f'torch_train/mlp_trained_weights_bias_correction_{battery}_complete_custom_loss.pth')
        trained_parameter_value = [mlp.cell.qMax.data.item(),mlp.cell.Ro.data.item()]
        print("Trained Parameter Value:", trained_parameter_value)
        if Validate_one:
            mlp.eval()
            # Time for the test set
            X = inputs_shiffed[val_idx,:,:]
            # For confidential reasons
            shape_X = np.shape(X)
            print(shape_X)

            Y = target_shiffed[val_idx,:,np.newaxis]  
            X_tensor = torch.from_numpy(X).to(DEVICE)
            Y_tensor = torch.from_numpy(Y).to(DEVICE)
            with torch.no_grad():
                pred = mlp(X_tensor).cpu().numpy()

            #plt.plot(X, Y, color='gray')
            print("Predictions have shape ",pred.shape)
            for i in range(NUM_CHECK):
                plt.plot(pred[i,:,0],linestyle='dashed')
                plt.plot(Y_tensor[i,:,0])
            plt.ylabel('Voltage(V)')
            plt.grid()

            plt.xlabel('Time')
            plt.savefig(f'figures/predictionvsreality_biascorrected_{battery}_complete_10_samples.png')
            plt.close()
        # Plot predictions
        #mlp.eval()
        #with torch.no_grad():
        #    pred = mlp(X_tensor).cpu().numpy()

        #plt.plot(X, Y, color='gray')
        #plt.plot(X, pred)
        #plt.grid()
        #plt.show()
    #plt.show()





    ###### FOR REFERENCE : TRAINING ENDS HERE #########        
    ###### This is a small experiment comparing warm_start with random start ########
    if EXPERIMENT:
        mlp = get_model(dt=dt, mlp=True, share_q_r=False, stateful=True,WARM_START=False).to(DEVICE)
        optimizer = optim.Adam(mlp.parameters(), lr=2e-2)
        criterion = nn.MSELoss().to(DEVICE)

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
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: 1 if epoch < 800 else (0.5 if epoch < 1100 else (0.25 if epoch < 2200 else 0.125)))

        # Training loop
        start = time.time()
        num_epochs = NUM_EPOCHS
        loss_cold_start = []
        mlp.train()
        for epoch in range(num_epochs):
            
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
            #scheduler.step()

            # Print epoch statistics
            if epoch % 100 == 0:
                loss_cold_start.append(total_loss / len(data_loader))
                print(f"Epoch {epoch}, Loss: {total_loss / len(data_loader)}, Time : {time.time()-start}")

        #Plot the two
        plt.plot(loss_warm_start)
        plt.plot(loss_cold_start)
        plt.ylabel('MSE(Loss)')
        plt.grid()

        plt.xlabel('Epoch(unit of 100)')
        plt.savefig('figures/trainingtrend.png')
        plt.show()
    ###### This is a small experiment comparing warm_start with random start ########
        
    ######## Validation is done here ##########
    if Validate:
        mlp.eval()
        # Time for the test set
        X = inputs_shiffed[val_idx,:,:]
        # For confidential reasons
        shape_X = np.shape(X)
        print(shape_X)

        Y = target_shiffed[val_idx,:,np.newaxis]  
        X_tensor = torch.from_numpy(X).to(DEVICE)
        Y_tensor = torch.from_numpy(Y).to(DEVICE)
        with torch.no_grad():
            pred = mlp(X_tensor).cpu().numpy()

        #plt.plot(X, Y, color='gray')
        print("Predictions have shape ",pred.shape)
        for i in range(NUM_CHECK):
            plt.plot(pred[i,:,0],linestyle='dashed')
            plt.plot(Y_tensor[i,:,0])
        plt.ylabel('Voltage(V)')
        plt.grid()

        plt.xlabel('Time')
        plt.savefig('figures/predictionvsreality_biascorrected.png')
        plt.show()

    ######## Validation is done here ##########

    ######### Just for Debugging Purposes #####
    debug = False
    if debug:

        mlp.eval()
        # Time for the test set
        X = inputs_shiffed[train_idx,:,:]
        # For confidential reasons
        shape_X = np.shape(X)
        #print(shape_X)

        Y = target_shiffed[train_idx,:,np.newaxis]  
        X_tensor = torch.from_numpy(X).to(DEVICE)
        Y_tensor = torch.from_numpy(Y).to(DEVICE)
        with torch.no_grad():
            pred = mlp(X_tensor).cpu().numpy()

        #plt.plot(X, Y, color='gray')
        print("Predictions have shape ",pred.shape)
        for i in range(NUM_CHECK):
            plt.plot(pred[i,:,0],linestyle='dashed')
            plt.plot(Y_tensor[i,:,0])
        plt.ylabel('Voltage(V)')
        plt.grid()

        plt.xlabel('Time')
        plt.savefig('figures/whatgoesonduringtraining.png')
        plt.show()
else:
    print("inputs_array has shape ",inputs_array.shape)
    for rows in inputs_array:
        plt.plot(rows)
    for rows in target_array:
        plt.plot(rows)
    plt.show()    



    ######### Just for Debugging Purposes #####