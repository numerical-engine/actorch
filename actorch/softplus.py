import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def psoftplus(x:torch.Tensor, a:float|torch.Tensor, b:float|torch.Tensor)->torch.Tensor:
    return a*(F.softplus(x) - b)

class PSoftplus(nn.Module):
    def __init__(self, a:float = 1.5, b:float = math.log(2), requires_grad:bool = False):
        super().__init__()
        self.a = nn.Parameter(torch.Tensor([a]), requires_grad = requires_grad)
        self.b = nn.Parameter(torch.Tensor([b]), requires_grad = requires_grad)
        
    def forward(self, x:torch.Tensor)->torch.Tensor:
        return psoftplus(x, self.a, self.b)

def softpp(x:torch.Tensor, a:float|torch.Tensor, b:float|torch.Tensor)->torch.Tensor:
    return torch.log(1. + torch.exp(a*x)) + x/b - math.log(2.)

class Softpp(nn.Module):
    def __init__(self, a:float = 1.5, b:float = math.log(2), requires_grad:bool = False):
        super().__init__()
        self.a = nn.Parameter(torch.Tensor([a]), requires_grad = requires_grad)
        self.b = nn.Parameter(torch.Tensor([b]), requires_grad = requires_grad)
        
    def forward(self, x:torch.Tensor)->torch.Tensor:
        return softpp(x, self.a, self.b)

def aranda_ordaz(x:torch.Tensor, a:float|torch.Tensor)->torch.Tensor:
    return 1. - torch.pow(1. + a*torch.exp(x), -1./a)

class Aranda_Ordaz(nn.Module):
    def __init__(self, a:float = 2., requires_grad:bool = False):
        super().__init__()
        self.a = nn.Parameter(torch.Tensor([a]), requires_grad = requires_grad)
        
    def forward(self, x:torch.Tensor)->torch.Tensor:
        return aranda_ordaz(x, self.a)

def pmaf(x:torch.Tensor, a:float|torch.Tensor)->torch.Tensor:
    return torch.where(x > 0.,
    2./math.sqrt(3.)*(math.pi**(-0.25))*(1. - (x - a)**2)*torch.exp(-0.5*(x - a)**2),
    2./math.sqrt(3.)*(math.pi**(-0.25))*(1. - (x + a)**2)*torch.exp(-0.5*(x + a)**2),
    )

class PMAF(nn.Module):
    def __init__(self, a:float = 4., requires_grad:bool = False):
        super().__init__()
        self.a = nn.Parameter(torch.Tensor([a]), requires_grad = requires_grad)
        
    def forward(self, x:torch.Tensor)->torch.Tensor:
        return pmaf(x, self.a)

def combhsine(x:torch.Tensor, a:float|torch.Tensor)->torch.Tensor:
    return torch.sinh(a*x) + 1./torch.sinh(a*x)

class CombHsine(nn.Module):
    def __init__(self, a:float, requires_grad:bool = False):
        super().__init__()
        self.a = nn.Parameter(torch.Tensor([a]), requires_grad = requires_grad)
        
    def forward(self, x:torch.Tensor)->torch.Tensor:
        return combhsine(x, self.a)

def marcsinh(x:torch.Tensor)->torch.Tensor:
    return torch.sqrt(x)/12./torch.sinh(x)

class MArsinh(nn.Module):        
    def forward(self, x:torch.Tensor)->torch.Tensor:
        return marcsinh(x)

def hypersinh(x:torch.Tensor)->torch.Tensor:
    return torch.where(x > 0., torch.sinh(x)/3., 0.25*x**3)

class HyperSinh(nn.Module):        
    def forward(self, x:torch.Tensor)->torch.Tensor:
        return hypersinh(x)