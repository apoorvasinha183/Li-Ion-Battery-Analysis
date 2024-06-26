import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from model import get_model


mlp = get_model(dt=10.0, mlp=True, share_q_r=False, stateful=True, version="naive_PINN").to(DEVICE)
optimizer = optim.Adam(mlp.parameters(), lr=2e-2)
criterion = nn.MSELoss()

# Prepare data
X = np.linspace(0.0, 1.0, 100).reshape(-1, 1).astype(np.float32)
X = np.expand_dims(X, axis=-1)
Y = np.hstack([np.linspace(0.85, -0.2, 95), np.linspace(-0.25, -0.8, 5)]).reshape(-1, 1).astype(np.float32)

# Convert data to PyTorch tensors
X_tensor = torch.from_numpy(X).to(DEVICE)
Y_tensor = torch.from_numpy(Y).to(DEVICE)

# Create PyTorch Dataset and DataLoader
dataset = TensorDataset(X_tensor, Y_tensor)
data_loader = DataLoader(dataset, batch_size=1, shuffle=True)

# Learning rate scheduler
scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: 3e-2 if epoch < 800 else (1e-2 if epoch < 1100 else (5e-3 if epoch < 2200 else 1e-3)))
print("Shapes are ",X_tensor.size(),Y_tensor.size())
# Training loop
num_epochs = 301
for epoch in range(num_epochs):
    mlp.train()
    total_loss = 0.0
    
    for inputs, targets in data_loader:
        optimizer.zero_grad()
        
        # Forward pass
        out = mlp(inputs)
        Volt, Vep, Ven, Vo, Vsn, Vsp = out.split(1, dim=1)

        # Compute loss
        loss = criterion(Volt, targets)

        
        # Backpropagation
        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    
    # Print epoch statistics
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {total_loss / len(data_loader)}")
    
    if epoch % 5 == 0:
        print(f"Loss is {loss}")
        print("Gradients after epoch", epoch)
        for name, param in mlp.named_parameters():
            if param.grad is not None:
                print(name, param.grad)

# Save model weights
PATH = 'torch_train/mlp_initial_weights_PINN.pth'
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

pred = pred[:,:,0]
X = X.squeeze(-1)
pred = pred[:, 0]
plt.plot(X, Y, color='gray')
plt.plot(X, pred)
plt.grid()
plt.show()
