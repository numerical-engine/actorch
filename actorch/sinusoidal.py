import torch
import torch.nn.functional as F
import torch.nn as nn
import math

def sin(x:torch.Tensor, a:float|torch.Tensor, b:float|torch.Tensor, c:float|torch.Tensor)->torch.Tensor:
    return a*torch.sin(b*x + c)

class Sin(nn.Module):
    def __init__(self, a:float = 1.0, b:float = 1.0, c:float = 0.0, requires_grad:bool = False)->None:
        super().__init__()
        self.a = nn.Parameter(torch.Tensor([a]), requires_grad = requires_grad)
        self.b = nn.Parameter(torch.Tensor([b]), requires_grad = requires_grad)
        self.c = nn.Parameter(torch.Tensor([c]), requires_grad = requires_grad)

    def forward(self, x:torch.Tensor)->torch.Tensor:
        return sin(x, self.a, self.b, self.c)

def snake(x:torch.Tensor, a:float)->torch.Tensor:
    return x + torch.pow(torch.sin(a*x), 2.0)

class Snake(nn.Module):
    def __init__(self, a:float = 2.0, requires_grad:bool = False)->None:
        super().__init__()
        self.a = nn.Parameter(torch.Tensor([a]), requires_grad = requires_grad)
    def forward(self, x:torch.Tensor)->torch.Tensor:
        return snake(x, self.a)


class STAF(nn.Module):
    def __init__(self, tau, omega_0=30)->None:
        super().__init__()
        self.ws = nn.Parameter(omega_0*torch.rand(tau), requires_grad=True)
        self.phis = nn.Parameter(-math.pi + 2 * math.pi * torch.rand(tau), requires_grad=True)
        diversity_y = 1 / (2 * tau)
        laplace_samples = torch.distributions.Laplace(0, diversity_y).sample((tau,))
        self.bs = nn.Parameter(torch.sign(laplace_samples) * torch.sqrt(torch.abs(laplace_samples)), requires_grad=True)
    
    def forward(self, x:torch.Tensor)->torch.Tensor:
        x = (self.bs * torch.sin(self.ws * x.unsqueeze(-1) + self.phis)).sum(dim = -1)
        return x

def finer(x:torch.Tensor)->torch.Tensor:
    return torch.sin((torch.abs(x) + 1.0)*x)

class FINER(nn.Module):
    def forward(self, x:torch.Tensor)->torch.Tensor:
        return finer(x)


def sinc(x:torch.Tensor, a:float|torch.Tensor)->torch.Tensor:
    return torch.where(x == 0., 1., torch.sin(a*x)/(a*x))

class Sinc(nn.Module):
    def __init__(self, a:float = math.pi, requires_grad:bool = False)->None:
        super().__init__()
        self.a = nn.Parameter(torch.Tensor([a]), requires_grad = requires_grad)
    
    def forward(self, x:torch.Tensor)->torch.Tensor:
        return sinc(x, self.a)

def sincsigmoid(x:torch.Tensor, a:float|torch.Tensor)->torch.Tensor:
    return sinc(F.sigmoid(x), a)

class SincSigmoid(Sinc):
    def forward(self, x:torch.Tensor)->torch.Tensor:
        return sinc_sigmoid(x, self.a)

def cosid(x:torch.Tensor)->torch.Tensor:
    return torch.cos(x) - x

class Cosid(nn.Module):
    def forward(self, x:torch.Tensor)->torch.Tensor:
        return cosid(x)

def sinp(x:torch.Tensor, a:float|torch.Tensor)->torch.Tensor:
    return torch.sin(x) - a*x

class Sinp(nn.Module):
    def __init__(self, a:float = 1.0, requires_grad:bool = False)->None:
        super().__init__()
        self.a = nn.Parameter(torch.Tensor([a]), requires_grad = requires_grad)
    
    def forward(self, x:torch.Tensor)->torch.Tensor:
        return sinp(x, self.a)

def gcu(x:torch.Tensor)->torch.Tensor:
    return x*torch.cos(x)

class GCU(nn.Module):
    def forward(self, x:torch.Tensor)->torch.Tensor:
        return gcu(x)

def asu(x:torch.Tensor)->torch.Tensor:
    return x*torch.sin(x)

class ASU(nn.Module):
    def forward(self, x:torch.Tensor)->torch.Tensor:
        return asu(x)

def dsu(x:torch.Tensor)->torch.Tensor:
    return 0.5*math.pi*(sinc(x - math.pi, math.pi) - sinc(x + math.pi, math.pi))

class DSU(nn.Module):
    def forward(self, x:torch.Tensor)->torch.Tensor:
        return dsu(x)

def hclsh(x:torch.Tensor)->torch.Tensor:
    return torch.where(x > 0.,
    torch.log(torch.cosh(x) + x*torch.cosh(x/2.0)),
    torch.log(torch.cosh(x)) + x
    )

class HcLSH(nn.Module):
    def forward(self, x:torch.Tensor)->torch.Tensor:
        return hclsh(x)

def css(x:torch.Tensor, a:float|torch.Tensor, b:float|torch.Tensor, c:float|torch.Tensor, d:float|torch.Tensor)->torch.Tensor:
    return a*torch.sin(b*x) + c*F.sigmoid(d*x)

class CSS(nn.Module):
    def __init__(self, a:float, b:float, c:float, d:float, requires_grad:bool = False)->None:
        super().__init__()
        self.a = nn.Parameter(torch.Tensor([a]), requires_grad = requires_grad)
        self.b = nn.Parameter(torch.Tensor([b]), requires_grad = requires_grad)
        self.c = nn.Parameter(torch.Tensor([c]), requires_grad = requires_grad)
        self.d = nn.Parameter(torch.Tensor([d]), requires_grad = requires_grad)
    
    def forward(self, x:torch.Tensor)->torch.Tensor:
        return css(x, self.a, self.b, self.c, self.d)

def expcos(x:torch.Tensor, a:float|torch.Tensor, b:float|torch.Tensor)->torch.Tensor:
    return torch.exp(-a*x**2)*torch.cos(b*x)

class Expcos(nn.Module):
    def __init__(self, a:float, b:float, requires_grad:bool = False)->None:
        super().__init__()
        self.a = nn.Parameter(torch.Tensor([a]), requires_grad = requires_grad)
        self.b = nn.Parameter(torch.Tensor([b]), requires_grad = requires_grad)
    
    def forward(self, x:torch.Tensor)->torch.Tensor:
        return expcos(x, self.a, self.b)