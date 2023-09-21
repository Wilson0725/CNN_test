# %%
### import 

# external 
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from torch import nn
from matplotlib import pyplot as plt

# internal 
from split import split
from model import famale_male_claasfer as FMClasser

# %% 
### setup

# path 
PATH = 'G://我的雲端硬碟//from NYCU address//For python//AI makeup advisor//train_set'

# device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'\n work on --{device}--\n')

# tensorboard
writer = SummaryWriter(log_dir='test_0910')

# transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),  
    transforms.ToTensor(),  
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  
])

# dataset 
batch_size = 64

dataset = datasets.ImageFolder(PATH, transform=transform)

data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# split
train_ratio = 0.8  # 80% 的數據用於訓練
test_ratio = 0.2   # 20% 的數據用於測試 

train_data, test_data = split(dataset,
                              train_ratio=train_ratio)

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

test_data_size = len(test_data)


# %% 
### bulit nn

# nn
model = FMClasser()
model.to(device)
print(f'\n{model}\n')

# loss 
loss_fn = nn.CrossEntropyLoss()
if torch.cuda.is_available():
    loss_fn = loss_fn.cuda()

# optim
learning_rate = 1e-3
weight_decay = 1e-5  # 加入正規化 適法避免過度擬和
optim = torch.optim.Adam(model.parameters(), lr=learning_rate,weight_decay=weight_decay)

# %%
### training process 

# record training step 
total_train_step = 0 

# recoed test step 
total_test_step = 0 

# epoch
epoch = 20

# record the loss trend 
train_loss_recrd = []
test_loss_recrd = [] 
accuracy_loss_recrd = []

for i in range(epoch):
    print(f'---------第{i+1}輪訓練開始-----------')
    
    # start to train 
    model.train()   # 有特殊層 dropout 甚麼的要調用 --> 其他層不影響

    for data in train_loader:
        images, targets = data

        # to device
        imgs = images.to(device)
        trgts = targets.to(device)

        # forward
        outputs = model(imgs)
        outputs.to(device)
        
        # loss
        loss = loss_fn(outputs, trgts)

    # apply optim 
    optim.zero_grad()
    loss.backward()
    optim.step()

    # +=1
    total_train_step += 1 
    print(f'訓練次數{total_train_step}, loss={loss.item()}') # .item不會帶 tensor format

    # record result 
    train_loss_recrd.append(loss.item())
    
    # tensorboard
    writer.add_scalar('train_loss', loss.item(), total_train_step)
# %%
#%%
### testing

    # start testing
    model.eval()   # 有特殊層 dropout 甚麼的要調用 --> 其他層不影響

    # test-phase started
    total_test_loss = 0

    # accuracy of classifier  
    total_accuracy = 0

    with torch.no_grad():
        for data in test_loader:
            imgs,targets =data

            # to device
            imgs = imgs.to(device)
            trgts = targets.to(device)

            outputs = model(imgs)
            outputs.to(device)

            if torch.cuda.is_available():
                imgs = imgs.cuda()
                outputs = outputs.cuda()
            loss = loss_fn(outputs, trgts)

            total_test_loss += loss.item()
            
            # accuracy
            accuracy = (outputs.argmax(1) == trgts).sum() # 總共對幾個
            total_accuracy += accuracy


    print(f"整體測試及上的 loss : {total_test_loss}") 
    print(f'整體測試集上的 accuracy: {total_accuracy/test_data_size}')
    
    # record result
    test_loss_recrd.append(total_test_loss)
    accuracy_loss_recrd.append(total_accuracy/test_data_size)
    
    # tensorboard 
    writer.add_scalar('test_loss_logs', total_test_loss, total_test_step)
    writer.add_scalar('test_accuracy_logs', total_accuracy, total_test_step)

    total_test_step += 1

    # save the model 
    #torch.save(model, f'cifar10_{i}_0904.pth')
    # torch.save(cifar10.state_dict(), f'cifar10_{i}_0904')
    #print("model saved!")

writer.close()

# %% 
### plotting result 

# fig, axs
fig, axs = plt.subplots(3, 1, figsize=(10, 15))

# train_loss
axs[0].plot(range(len(train_loss_recrd)), train_loss_recrd, label='Train Loss', color='blue')
axs[0].set_xlabel('Training Steps')
axs[0].set_ylabel('Loss')
axs[0].set_title('Training Loss')
axs[0].legend()
axs[0].grid(True)

# test_loss
axs[1].plot(range(len(test_loss_recrd)), test_loss_recrd, label='Test Loss', color='red')
axs[1].set_xlabel('Testing Steps')
axs[1].set_ylabel('Loss')
axs[1].set_title('Testing Loss')
axs[1].legend()
axs[1].grid(True)

# acurracy 
accuracy_loss_recrd_cpu = [acc.cpu().numpy() for acc in accuracy_loss_recrd]  # cuda to CPU
axs[2].plot(range(len(accuracy_loss_recrd_cpu)), accuracy_loss_recrd_cpu, label='Test Accuracy', color='green')
axs[2].set_xlabel('Testing Steps')
axs[2].set_ylabel('Accuracy')
axs[2].set_title('Testing Accuracy')
axs[2].legend()
axs[2].grid(True)

plt.tight_layout()
plt.show()