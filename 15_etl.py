import torch
import numpy as np
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
train_set=torchvision.datasets.FashionMNIST(root='./data',
                                   train= True,
                                   download=True,
                                   transform=transforms.Compose([transforms.ToTensor()]))
train_loader=torch.utils.data.DataLoader(train_set, batch_size=10)
torch.set_printoptions(linewidth=120)
print('length of train_set:',len(train_set))
print('labels:',train_set.targets)
print('frequency:',train_set.targets.bincount())#各类别分布均匀60000=10*6000
#加载单个样本图片
sample=next(iter(train_set))#len(sample)==2,每个样本包含图像和label,[image(matrix),label]
print(type(sample))#list,tuple(元组)都是可迭代对象
image,label,=sample
print('shape of image:',image.shape)#image为单通道的28*28
plt.imshow(image.squeeze(),cmap='gray')
plt.title(label)
#加载一批图片(batch)
display_loader=torch.utils.data.DataLoader(train_set,batch_size=10)
batch=next(iter(display_loader))
print('length of batch:',len(batch))
images,labels=batch
print('image size:',images.shape)#images size: torch.Size([10, 1, 28, 28])
grid=torchvision.utils.make_grid(images,nrow=10)#torch.Size([3, 32, 302])
plt.figure(figsize=[10,10])
plt.imshow(np.transpose(grid, (1,2,0)))#imshow只能画二维，或[x,y,3/4]的matrix
plt.show()