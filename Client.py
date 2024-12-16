import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torchvision
import random

class Client:
    def __init__(self, model, dataloader, lr=0.01, local_epochs=5, mu=1e4):

        self.model = model
        self.dataloader = dataloader
        self.lr = lr
        self.local_epochs = local_epochs
        self.mu = mu  # fedprox regularization term

    def synchronize(self, global_state_dict):
        self.model.load_state_dict(global_state_dict)

    def train(self, global_state_dict):

        self.model.train()
        optimizer = optim.SGD(self.model.parameters(), lr=self.lr)
        loss_fn = nn.CrossEntropyLoss()

        global_params = {name: param.clone().detach() for name, param in global_state_dict.items()}

        for epoch in range(self.local_epochs):  
            for X, y in self.dataloader:
                optimizer.zero_grad()
                outputs = self.model(X)

                loss = loss_fn(outputs, y)

                proximal_term = 0.0
                for name, param in self.model.named_parameters():
                    proximal_term += torch.sum((param - global_params[name]) ** 2)
                loss += (self.mu / 2) * proximal_term  # fedbrox loss

                loss.backward()
                optimizer.step()

        return self.model.state_dict()
