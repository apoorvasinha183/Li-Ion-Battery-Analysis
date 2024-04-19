# This is for handling the RNN Data
from torch.utils.data import DataLoader

# Assuming `dataset` is your SequentialDataset instance
batch_size = 32
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
