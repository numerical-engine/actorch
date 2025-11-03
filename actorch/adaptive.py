import torch
import torch.nn.functional as F
import torch.nn as nn
import math

class DKNN(nn.Module):
    def __init__(self, num:int, act_funcs:list[any])->None:
        assert num == len(act_funcs)
        super().__init__()
        self.act_funcs = act_funcs
        self.alpha = torch.zeros(num)
        self.alpha[0] = 1.
        self.alpha = nn.Parameter(self.alpha, requires_grad = True)
        
        self.omega = nn.Parameter(torch.ones(num), requires_grad = True)
    
    def forward(self, x:torch.Tensor)->torch.Tensor:
        return torch.sum(torch.stack([a*act(w*x) for a, w, act in zip(self.alpha, self.omega, self.act_funcs)]), dim = 0)

class TAAF(nn.Module):
    def __init__(self, shape:int|tuple, act:any)->None:
        super().__init__()
        self.act = act
        self.alpha = nn.Parameter(torch.ones(shape), requires_grad = True)
        self.beta = nn.Parameter(torch.ones(shape), requires_grad = True)
        self.gamma = nn.Parameter(torch.zeros(shape), requires_grad = True)
        self.delta = nn.Parameter(torch.zeros(shape), requires_grad = True)
        self.act = act
    def forward(self, x:torch.Tensor)->torch.Tensor:
        return self.alpha*self.act(self.beta*x + self.gamma) + self.delta

class PReLU_nodewise(nn.Module):
    def __init__(self, shape:int|tuple)->None:
        super().__init__()
        self.a = nn.Parameter(0.25*torch.ones(shape), requires_grad = True)
    
    def forward(self, x:torch.Tensor)-> torch.Tensor:
        return torch.where(x > 0., x, self.a*x)

class PReLUp_nodewise(nn.Module):
    def __init__(self, shape:int|tuple)->None:
        super().__init__()
        self.a = nn.Parameter(torch.ones(shape), requires_grad = True)
    
    def forward(self, x:torch.Tensor)-> torch.Tensor:
        return torch.where(x > 0., self.a*x, 0.)

class PREU(nn.Module):
    def __init__(self, shape:int|tuple)->None:
        super().__init__()
        self.a = nn.Parameter(torch.ones(shape), requires_grad = True)
        self.b = nn.Parameter(torch.ones(shape), requires_grad = True)
    
    def forward(self, x:torch.Tensor)->torch.Tensor:
        return torch.where(x > 0., a*x, a*x*torch.exp(b*x))

class FReLU(nn.Module):
    def __init__(self, shape:int|tuple)->None:
        super().__init__()
        self.a = nn.Parameter(torch.zeros(shape), requires_grad = True)
    
    def forward(self, x:torch.Tensor)->torch.Tensor:
        return F.relu(x) + self.a

class StarReLU(nn.Module):
    def __init__(self, shape:int|tuple)->None:
        super().__init__()
        self.a = nn.Parameter(0.8944*torch.ones(shape), requires_grad = True)
        self.b = nn.Parameter(-0.4472*torch.ones(shape), requires_grad = True)
    
    def forward(self, x:torch.Tensor)->torch.Tensor:
        return self.a*F.relu(x)**2 + self.b

class DualPReLU(nn.Module):
    def __init__(self, shape:int|tuple)->None:
        super().__init__()
        self.a = nn.Parameter(torch.ones(shape), requires_grad = True)
        self.b = nn.Parameter(0.01*torch.ones(shape), requires_grad = True)
    
    def forward(self, x:torch.Tensor)->torch.Tensor:
        return torch.where(x > 0., self.a*x, self.b*x)

class FPAF(nn.Module):
    def __init__(self, shape:int|tuple, act_funcs:list[any, any])->None:
        super().__init__()
        self.a = nn.Parameter(torch.ones(shape), requires_grad = True)
        self.b = nn.Parameter(torch.ones(shape), requires_grad = True)
        self.f, self.g = act_funcs

    def forward(self, x:torch.Tensor)->torch.Tensor:
        return torch.where(x > 0., self.a*self.f(x), self.b*self.g(x))

class PTELU(nn.Module):
    def __init__(self, shape:int|tuple)->None:
        super().__init__()
        self.a = nn.Parameter(torch.ones(shape), requires_grad = True)
        self.b = nn.Parameter(torch.ones(shape), requires_grad = True)
    
    def forward(self, x:torch.Tensor)->torch.Tensor:
        return torch.where(x > 0., x, self.a*F.tanh(self.b*x))

class TanhLU(nn.Module):
    def __init__(self, shape:int|tuple)->None:
        super().__init__()
        self.a = nn.Parameter(torch.ones(shape), requires_grad = True)
        self.b = nn.Parameter(torch.ones(shape), requires_grad = True)
        self.c = nn.Parameter(torch.zeros(shape), requires_grad = True)
    
    def forward(self, x:torch.Tensor)->torch.Tensor:
        return self.a*F.tanh(self.b*x) + self.c*x

class BLU(nn.Module):
    def __init__(self, shape:int|tuple)->None:
        super().__init__()
        self.a = nn.Parameter(torch.zeros(shape), requires_grad = True)
    
    def forward(self, x:torch.Tensor)->torch.Tensor:
        return F.sigmoid(self.a)*(torch.sqrt(x**2 + 1.) - 1.) + x

class ReBLU(BLU):
    def forward(self, x:torch.Tensor)->torch.Tensor:
        z = super().forward(x)
        return torch.where(x > 0., z, 0.)