import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.imitation_learning.core.dataset import HDF5Dataset
from src.imitation_learning.networks.mlp import PolicyNetwork

dataset = HDF5Dataset(
        "/mnt/Daten/102_Master/70_Thesis/711_CodeRepo/bimanual-imitation-learning/demonstrations/wp/Panda_Panda_TwoArmPickPlace_2025-01-02_17-10-41/demo.hdf5",
)

train, test = dataset.get_splits()

train_loader = DataLoader(train, batch_size=32, shuffle=True)
test_loader = DataLoader(test, batch_size=32, shuffle=True)

policy_net = PolicyNetwork(np.prod(dataset.input_size), np.prod(dataset.output_size))
optimizer = torch.optim.Adam(policy_net.parameters())
loss_fn = nn.MSELoss()

num_epochs = 32

for epoch in range(num_epochs):
    for obs, action in train_loader:
        action_pred = policy_net(obs)
        loss = loss_fn(action_pred, action)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # TODO add evaluation

    # TODO add logging

    print(f"Epoch: {epoch}/{num_epochs}, Loss: {loss.item()}")