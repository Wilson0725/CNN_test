# %%
### import 

# external 
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

def split(dataset,train_ratio):
    num_data = len(dataset)
    num_train = int(train_ratio * num_data)
    num_test = num_data - num_train

    train_data, test_data = random_split(dataset, [num_train, num_test])
    
    
    return train_data, test_data