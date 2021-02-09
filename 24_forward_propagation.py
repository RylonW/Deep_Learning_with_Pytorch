import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision
import torchvision.transforms as transforms
torch.set_printoptions(linewidth=120)
train_set=torchvision.datasets.FashionMNIST(root='./data',
                                            train= True,
                                            download=True,
                                            transform=transforms.Compose([transforms.ToTensor()]))
class Network(nn.Module):
    def __init__(self):
        super().__init__()#super()调用父类
        self.conv1=nn.Conv2d(in_channels=1,out_channels=6,kernel_size=5)
        self.conv2=nn.Conv2d(in_channels=6,out_channels=12,kernel_size=5)

        self.fc1=nn.Linear(in_features=12*4*4,out_features=120)
        self.fc2=nn.Linear(in_features=120,out_features=60)
        self.out=nn.Linear(in_features=60,out_features=10)

    def forward(self,t):
        t=F.relu(self.conv1(t))
        t=F.max_pool2d(t,kernel_size=2,stride=2)

        t=F.relu(self.conv2(t))
        t=F.max_pool2d(t,kernel_size=2,stride=2)

        t=F.relu(self.fc1(t.reshape(-1,12*4*4)))
        t=F.relu(self.fc2(t))
        t=self.out(t)

        return t
torch.set_grad_enabled(False)
network=Network()
sample=next(iter(train_set))
image,label=sample
print(image.shape)
print(image.unsqueeze(0).shape)#增加维度
pred=network(image.unsqueeze(0))
print('predp.shape',pred.shape)
print('pred_label:',pred.argmax(dim=1),'\n','label:',label)