import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Mimics mlp_tests.py in TF
# Let batch size be 1 
#This seems slower
# Define the MLP model
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

# Create the MLP model, optimizer, and criterion
mlp = MLP().to(DEVICE)
optimizer = optim.Adam(mlp.parameters(), lr=2e-2)
criterion = nn.MSELoss()

# Prepare data
X = np.linspace(0.0, 1.0, 100).reshape(-1, 1).astype(np.float32)
Y = np.hstack([np.linspace(0.85, -0.2, 90), np.linspace(-0.25, -0.8, 10)]).reshape(-1, 1).astype(np.float32)

# Convert data to PyTorch tensors
X_tensor = torch.from_numpy(X).to(DEVICE)
Y_tensor = torch.from_numpy(Y).to(DEVICE)

# Create PyTorch Dataset and DataLoader
dataset = TensorDataset(X_tensor, Y_tensor)
data_loader = DataLoader(dataset, batch_size=32, shuffle=True)

# Learning rate scheduler
#scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: 2e-2 if epoch < 800 else (1e-2 if epoch < 1100 else (5e-3 if epoch < 2200 else 1e-3)))
print("Shapes are ",X_tensor.size(),Y_tensor.size())
# Training loop
num_epochs = 10000
mlp.train()
optimizer = optim.Adam(mlp.parameters(), lr=2e-2)
for epoch in range(num_epochs):
    
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
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {total_loss / len(data_loader)}")
        for param_group in optimizer.param_groups:
            current_learning_rate = param_group['lr']
            print("Current Learning Rate:", current_learning_rate)

# Save model weights
PATH = 'torch_train/mlp_initial_weights.pth'
torch.save({
    'epoch': epoch,
    'model_state_dict': mlp.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': loss
    }, PATH)
#torch.save(mlp.state_dict(), 'torch_train/mlp_initial_weights.pth')

# Plot predictions
mlp.eval()
with torch.no_grad():
    pred = mlp(X_tensor).cpu().numpy()

#plt.plot(X, Y, color='gray')
#plt.plot(X, pred)
#plt.grid()
#plt.show()
