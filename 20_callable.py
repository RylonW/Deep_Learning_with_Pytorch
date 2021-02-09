import torch
import torch.nn as nn
in_features=torch.tensor([1,2,3,4],dtype=torch.float32)
weights=torch.tensor([[1,2,3,4],
                      [2,3,4,5],
                      [3,4,5,6]],dtype=torch.float32)
output=weights.matmul(in_features)
print(output)
fc=nn.Linear(in_features=4,out_features=3,bias=False)
fc.weight=nn.Parameter(weights)
feature_out=fc(in_features)
print(feature_out)#bias=True tensor([30.0376, 40.3309, 49.5440], grad_fn=<AddBackward0>)
