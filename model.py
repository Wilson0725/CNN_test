# %%
### import 

# external 
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from torch import nn
import torch

# %% 

class famale_male_claasfer(nn.Module):
    
    def __init__(self):
        super(famale_male_claasfer,self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3,32,5,1,2),
            nn.MaxPool2d(2),
            nn.Conv2d(32,32,5,1,2),
            nn.MaxPool2d(2),
            nn.Conv2d(32,64,5,1,2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64*28*28,64),
            nn.Linear(64,2),
            nn.Softmax(dim=1) 
        )

        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.model(x)
        x = self.dropout(x)
        return x
        