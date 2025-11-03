import torch
import torch.nn as nn
import torch.nn.functional as F

def swiglu(x1:torch.Tensor, x2:torch.Tensor, beta:float|torch.Tensor)->torch.Tensor:
    return x1*F.sigmoid(beta*x1)*x2

class SwiGLU(nn.Module):
    def __init__(self, beta:float = 1.0)->None:
        super().__init__()
        self.beta = nn.Parameter(torch.tensor([beta]), requires_grad = True)
    
    def forward(self, x1:torch.Tensor, x2:torch.Tensor)->torch.Tensor:
        return swiglu(x1, x2, self.beta)


def reglu(x1:torch.Tensor, x2:torch.Tensor)->torch.Tensor:
    return x1*F.relu(x2)

class ReGLU(nn.Module):
    def forward(self, x1:torch.Tensor, x2:torch.Tensor)->torch.Tensor:
        return reglu(x1, x2)

def geglu(x1:torch.Tensor, x2:torch.Tensor)->torch.Tensor:
    return x1*F.gelu(x2)

class GEGLU(nn.Module):
    def forward(self, x1:torch.Tensor, x2:torch.Tensor)->torch.Tensor:
        return geglu(x1, x2)

def gtu(x1:torch.Tensor, x2:torch.Tensor)->torch.Tensor:
    return x1*F.tanh(x2)

class GTU(nn.Module):
    def forward(self, x1:torch.Tensor, x2:torch.Tensor)->torch.Tensor:
        return gtu(x1, x2)