import torch.nn as nn
class Netowrk:
    def __init__(self):
        self.layer=None
    def forward(self,t):
        t=self.layer(t)
        return t
class Network(nn.Module):
    def __init__(self):
        super(Network,self).__init__()#super()调用父类
        self.conv1=nn.Conv2d(in_channels=1,out_channels=6,kernel_size=5)
        self.conv2=nn.Conv2d(in_channels=6,out_channels=12,kernel_size=5)

        self.fc1=nn.Linear(in_features=12*4*4,out_features=120)
        self.fc2=nn.Linear(in_features=120,out_features=60)
        self.out=nn.Linear(in_features=60,out_features=10)

    def forward(self,t):
        return t
network=Network()
for name,param in network.named_parameters():
    print(name,'\t',param.shape)
#bias可训练
