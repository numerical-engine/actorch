import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def gauss(x:torch.Tensor, sigma:float|torch.Tensor)->torch.Tensor:
    return torch.exp(-(sigma*x)**2)

class Gauss(nn.Module):
    def __init__(self, sigma:float = 10, requires_grad:bool = False):
        super().__init__()
        self.sigma = nn.Parameter(torch.Tensor([sigma]), requires_grad = requires_grad)
        
    def forward(self, x:torch.Tensor)->torch.Tensor:
        return gauss(x, self.sigma)

def isrlu(x:torch.Tensor, a:float|torch.Tensor)->torch.Tensor:
    return torch.where(x > 0., x, x/torch.sqrt(1. + a*x**2))

class ISRLU(nn.Module):
    def __init__(self, a:float = 1.0, requires_grad:bool = False)->None:
        super().__init__()
        self.a = nn.Parameter(torch.Tensor([a]), requires_grad = requires_grad)
    
    def forward(self, x:torch.Tensor)->torch.Tensor:
        return isrlu(x, self.a)

def isru(x:torch.Tensor)->torch.Tensor:
    return x/torch.sqrt(1. + a*x**2)

class ISRU(nn.Module):
    def __init__(self, a:float = 1.0, requires_grad:bool = False)->None:
        super().__init__()
        self.a = nn.Parameter(torch.Tensor([a]), requires_grad = requires_grad)
    
    def forward(self, x:torch.Tensor)->torch.Tensor:
        return isru(x, self.a)

def mef(x:torch.Tensor)->torch.Tensor:
    return x/torch.sqrt(1. + x**2) + 0.5

class MEF(nn.Module):
    def forward(self, x:torch.Tensor)->torch.Tensor:
        return mef(x)

def blu(x:torch.Tensor)->torch.Tensor:
    return (torch.sqrt(x**2 + 1.) - 1.)/2. + z

class BLU(nn.Module):
    def forward(self, x:torch.Tensor)->torch.Tensor:
        return blu(x)

def mishra(x:torch.Tensor)->torch.Tensor:
    return 0.5*torch.pow(x/(1. + torch.abs(x)), 2.) + 0.5*x/(1. + torch.abs(x))

class Mishra(nn.Module):
    def forward(self, x:torch.Tensor)->torch.Tensor:
        return mishra(x)

def sbaf(x:torch.Tensor, a:float|torch.Tensor, b:torch.Tensor)->torch.Tensor:
    return 1./(1. + b*torch.pow(x, a)*torch.pow(1. - x, 1. - a))

class SBAF(nn.Module):
    def __init__(self, a:float = 0.5, b:float = 0.98, requires_grad:bool = False)->None:
        super().__init__()
        self.a = nn.Parameter(torch.Tensor([a]), requires_grad = requires_grad)
        self.b = nn.Parameter(torch.Tensor([b]), requires_grad = requires_grad)
    
    def forward(self, x:torch.Tensor)->torch.Tensor:
        return sbaf(x, self.a, self.b)

def laf(x:torch.Tensor)->torch.Tensor:
    return torch.where(x > 0., torch.log(x) + 1., -torch.log(-x) + 1.)

class LAF(nn.Module):
    def forward(self, x:torch.Tensor)->torch.Tensor:
        return laf(x)

def symexp(x:torch.Tensor)->torch.Tensor:
    return torch.sgn(x)*(torch.exp(torch.abs(x)) - 1.)
class Symexp(nn.Module):
    def forward(self, x:torch.Tensor)->torch.Tensor:
        return symexp(x)

def etanh(x:torch.Tensor, a:float|torch.Tensor)->torch.Tensor:
    return a*torch.exp(x)*F.tanh(x)

class ETanh(nn.Module):
    def __init__(self, a:float, requires_grad:bool = False)->None:
        super().__init__()
        self.a = nn.Parameter(torch.Tensor([a]), requires_grad = requires_grad)
    
    def forward(self, x:torch.Tensor)->torch.Tensor:
        return etanh(x, self.a)

def wave(x:torch.Tensor, a:float|torch.Tensor)->torch.Tensor:
    return (1. - x**2)*torch.exp(-a*x**2)

class Wave(nn.Module):
    def __init__(self, a:float, requires_grad:bool = False)->None:
        super().__init__()
        self.a = nn.Parameter(torch.Tensor([a]), requires_grad = requires_grad)
    
    def forward(self, x:torch.Tensor)->torch.Tensor:
        return wave(x, self.a)

def eswish(x:torch.Tensor, a:float|torch.Tensor)->torch.Tensor:
    return a*x*F.sigmoid(x)

class ESwish(nn.Module):
    def __init__(self, a:float = 1.5, requires_grad:bool = False)->None:
        super().__init__()
        self.a = nn.Parameter(torch.Tensor([a]), requires_grad = requires_grad)
    
    def forward(self, x:torch.Tensor)->torch.Tensor:
        return eswish(x, self.a)

def naf(x:torch.Tensor, a:float|torch.Tensor, b:float|torch.Tensor, c:float|torch.Tensor, d:float|torch.Tensor)->torch.Tensor:
    return a*torch.exp(-b*x**2) + c/(1. + torch.exp(-d*x))

class NAF(nn.Module):
    def __init__(self, a:float, b:float, c:float, d:float, requires_grad:bool = False)->None:
        super().__init__()
        self.a = nn.Parameter(torch.Tensor([a]), requires_grad = requires_grad)
        self.b = nn.Parameter(torch.Tensor([b]), requires_grad = requires_grad)
        self.c = nn.Parameter(torch.Tensor([c]), requires_grad = requires_grad)
        self.d = nn.Parameter(torch.Tensor([d]), requires_grad = requires_grad)
    
    def forward(self, x:torch.Tensor)->torch.Tensor:
        return naf(x, self.a, self.b, self.c, self.d)