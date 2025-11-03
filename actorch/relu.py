import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def slrelu(x:torch.Tensor, a:float|torch.Tensor)->torch.Tensor:
    return torch.where(x > 0., a*x, 0.)

class SlReLU(nn.Module):
    def __init__(self, a:float = 1.0, requires_grad:bool = False)->None:
        super().__init__()
        self.a = nn.Parameter(torch.Tensor([a]), requires_grad = requires_grad)
    
    def forward(self, x:torch.Tensor)->torch.Tensor:
        return slrelu(x, self.a)

def sinerelu(x:torch.Tensor, a:float|torch.Tensor)->torch.Tensor:
    return torch.where(x > 0., x, a*(torch.sin(x)-torch.cos(x)))

class SineReLU(nn.Module):
    def __init__(self, a:float = 1.0, requires_grad:bool = False)->None:
        super().__init__()
        self.a = nn.Parameter(torch.Tensor([a]), requires_grad = requires_grad)
    
    def forward(self, x:torch.Tensor)->torch.Tensor:
        return sinerelu(x, self.a)

def minsin(x:torch.Tensor)->torch.Tensor:
    return torch.where(x > 0., torch.sin(x), x)

class Minsin(nn.Module):
    def forward(self, x:torch.Tensor)->torch.Tensor:
        return minsin(x)

def vlu(x:torch.Tensor, a:float|torch.Tensor, b:float|torch.Tensor)->torch.Tensor:
    return F.relu(x) + a*torch.sin(b*x)

class VLU(nn.Module):
    def __init__(self, a:float = 1.0, b:float = 1.0, requires_grad:bool = False)->None:
        super().__init__()
        self.a = nn.Parameter(torch.Tensor([a]), requires_grad = requires_grad)
        self.b = nn.Parameter(torch.Tensor([b]), requires_grad = requires_grad)
    
    def forward(self, x:torch.Tensor)->torch.Tensor:
        return vlu(x, self.a, self.b)

def nlrelu(x:torch.Tensor, a:float|torch.Tensor)->torch.Tensor:
    return torch.log(a*F.relu(x) + 1.0)

class NLReLU(nn.Module):
    def __init__(self, a:float = 1.0, requires_grad:bool = False)->None:
        super().__init__()
        self.a = nn.Parameter(torch.Tensor([a]), requires_grad = requires_grad)
    def forward(self, x:torch.Tensor)->torch.Tensor:
        return nlrelu(x, self.a)

def slu(x:torch.Tensor, a:float|torch.Tensor, b:float|torch.Tensor, c:float|torch.Tensor)->torch.Tensor:
    return torch.where(x > 0., a*x, b*torch.log(torch.exp(x) + 1.) - c)

class SLU(nn.Module):
    def __init__(self, a:float = 1.0, b:float = 2.0, c:float = 2.0*math.log(2.0), requires_grad:bool = False)->None:
        super().__init__()
        self.a = nn.Parameter(torch.Tensor([a]), requires_grad = requires_grad)
        self.b = nn.Parameter(torch.Tensor([b]), requires_grad = requires_grad)
        self.c = nn.Parameter(torch.Tensor([c]), requires_grad = requires_grad)

    def forward(self, x:torch.Tensor)->torch.Tensor:
        return slu(x, self.a, self.b, self.c)

def resp(x:torch.Tensor, a:float|torch.Tensor)->torch.Tensor:
    return torch.where(x > 0., a*x + math.log(2.), torch.log(1. + torch.exp(x)))

class ReSP(nn.Module):
    def __init__(self, a:float = 1.7, requires_grad:bool = False)->None:
        super().__init__()
        self.a = nn.Parameter(torch.Tensor([a]), requires_grad = requires_grad)

    def forward(self, x:torch.Tensor)->torch.Tensor:
        return slu(x, self.a, self.b, self.c)

def prenu(x:torch.Tensor, a:float|torch.Tensor)->torch.Tensor:
    return torch.where(x > 0., x - a*torch.log(x + 1.0), 0.)

class PReNU(nn.Module):
    def __init__(self, a:float = 1.0, requires_grad:bool = False)->None:
        super().__init__()
        self.a = nn.Parameter(torch.Tensor([a]), requires_grad = requires_grad)

    def forward(self, x:torch.Tensor)->torch.Tensor:
        return prenu(x, self.a)

def repu(x:torch.Tensor, a:float|torch.Tensor)->torch.Tensor:
    return torch.where(x > 0., x**a, 0.)

class RePU(nn.Module):
    def __init__(self, a:float = 1.0, requires_grad:bool = False)->None:
        super().__init__()
        self.a = nn.Parameter(torch.Tensor([a]), requires_grad = requires_grad)

    def forward(self, x:torch.Tensor)->torch.Tensor:
        return repu(x, self.a)

def fts(x:torch.Tensor, a:float|torch.Tensor)->torch.Tensor:
    return F.relu(x)*F.sigmoid(x) + a

class FTS(nn.Module):
    def __init__(self, a:float = -0.2, requires_grad:bool = False)->None:
        super().__init__()
        self.a = nn.Parameter(torch.Tensor([a]), requires_grad = requires_grad)

    def forward(self, x:torch.Tensor)->torch.Tensor:
        return fts(x, self.a)

def oaf(x:torch.Tensor)->torch.Tensor:
    return F.relu(x) + x*F.sigmoid(x)

class OAF(nn.Module):
    def forward(self, x:torch.Tensor)->torch.Tensor:
        return oaf(x)

def reu(x:torch.Tensor)->torch.Tensor:
    return torch.where(x > 0., x, x*torch.exp(x))

class REU(nn.Module):
    def forward(self, x:torch.Tensor)->torch.Tensor:
        return reu(x)

def ada(x:torch.Tensor, a:float|torch.Tensor, b:float|torch.Tensor)->torch.Tensor:
    return torch.where(x > 0., torch.exp(-a*x+b), 0.)

class ADA(nn.Module):
    def __init__(self, a:float, b:float, requires_grad:bool = False)->None:
        super().__init__()
        self.a = nn.Parameter(torch.Tensor([a]), requires_grad = requires_grad)
        self.b = nn.Parameter(torch.Tensor([b]), requires_grad = requires_grad)

    def forward(self, x:torch.Tensor)->torch.Tensor:
        return ada(x, self.a, self.b)

def lada(x:torch.Tensor, a:float|torch.Tensor, b:float|torch.Tensor, c:float|torch.Tensor)->torch.Tensor:
    return torch.where(x > 0., torch.exp(-a*x+b), c*x)

class LADA(nn.Module):
    def __init__(self, a:float, b:float, c:float = 0.01, requires_grad:bool = False)->None:
        super().__init__()
        self.a = nn.Parameter(torch.Tensor([a]), requires_grad = requires_grad)
        self.b = nn.Parameter(torch.Tensor([b]), requires_grad = requires_grad)
        self.c = nn.Parameter(torch.Tensor([c]), requires_grad = requires_grad)

    def forward(self, x:torch.Tensor)->torch.Tensor:
        return lada(x, self.a, self.b, self.c)

def siglu(x:torch.Tensor)->torch.Tensor:
    return torch.where(x > 0., x, (1. - torch.exp(-2*x))/(1. + torch.exp(-2*x)))

class SigLU(nn.Module):
    def forward(self, x:torch.Tensor)->torch.Tensor:
        return siglu(x)

def sara(x:torch.Tensor, a:float|torch.Tensor, b:float|torch.Tensor)->torch.Tensor:
    return torch.where(x > 0., x, x/(1. + a*torch.exp(-b*x)))

class SaRa(nn.Module):
    def __init__(self, a:float = 0.5, b:float = 0.7, requires_grad:bool = False)->None:
        super().__init__()
        self.a = nn.Parameter(torch.Tensor([a]), requires_grad = requires_grad)
        self.b = nn.Parameter(torch.Tensor([b]), requires_grad = requires_grad)

    def forward(self, x:torch.Tensor)->torch.Tensor:
        return sara(x, self.a, self.b)

def maxsig(x:torch.Tensor)->torch.Tensor:
    return torch.max(x, F.sigmoid(x))

class Maxsig(nn.Module):
    def forward(self, x:torch.Tensor)->torch.Tensor:
        return maxsig(x)

def thlu(x:torch.Tensor)->torch.Tensor:
    return torch.where(x > 0., x, F.tanh(x/2.))

class ThLU(nn.Module):
    def forward(self, x:torch.Tensor)->torch.Tensor:
        return thlu(x)

def dualelu(x1:torch.Tensor, x2:torch.Tensor)->torch.Tensor:
    return F.elu(x1) - F.elu(x2)

class DualELU(nn.Module):
    def forward(self, x1:torch.Tensor, x2:torch.Tensor)->torch.Tensor:
        return thlu(x1, x2)

def diffelu(x:torch.Tensor, a:float|torch.Tensor, b:float|torch.Tensor)->torch.Tensor:
    return torch.where(x > 0., x, a*(x*torch.exp(x)-b*torch.exp(b*x)))

class DiffELU(nn.Module):
    def __init__(self, a:float = 0.3, b:float = 0.1, requires_grad:bool = False)->None:
        super().__init__()
        self.a = nn.Parameter(torch.Tensor([a]), requires_grad = requires_grad)
        self.b = nn.Parameter(torch.Tensor([b]), requires_grad = requires_grad)

    def forward(self, x:torch.Tensor)->torch.Tensor:
        return diffelu(x, self.a, self.b)

def polylu(x:torch.Tensor)->torch.Tensor:
    return torch.where(x > 0., x, 1./(1. - x) - 1.)

class DualELU(nn.Module):
    def forward(self, x)->torch.Tensor:
        return polylu(x)

def iplu(x:torch.Tensor, a:float|torch.Tensor)->torch.Tensor:
    return torch.where(x > 0., x, 1./(1. + torch.abs(x)**a))

class IpLU(nn.Module):
    def __init__(self, a:float, requires_grad:bool = False)->None:
        super().__init__()
        self.a = nn.Parameter(torch.Tensor([a]), requires_grad = requires_grad)

    def forward(self, x:torch.Tensor)->torch.Tensor:
        return iplu(x, self.a)

def polu(x:torch.Tensor, a:float|torch.Tensor)->torch.Tensor:
    return torch.where(x > 0., x, torch.pow(1. - x, -a) - 1.)

class PoLU(nn.Module):
    def __init__(self, a:float = 1.5, requires_grad:bool = False)->None:
        super().__init__()
        self.a = nn.Parameter(torch.Tensor([a]), requires_grad = requires_grad)

    def forward(self, x:torch.Tensor)->torch.Tensor:
        return polu(x, self.a)

def pflu(x:torch.Tensor)->torch.Tensor:
    return 0.5*x*(1. + x/torch.sqrt(1. + x**2))

class PFLU(nn.Module):
    def forward(self, x:torch.Tensor)->torch.Tensor:
        return pflu(x)

def fpflu(x:torch.Tensor)->torch.Tensor:
    return torch.where(x > 0., x, x + torch.pow(x, 2.)/torch.sqrt(1. + torch.pow(x, 2.)))

class FPFLU(nn.Module):
    def forward(self, x:torch.Tensor)->torch.Tensor:
        return fpflu(x)

def lselu(x:torch.Tensor, a:float|torch.Tensor, b:float|torch.Tensor, c:float|torch.Tensor)->torch.Tensor:
    return torch.where(x > 0., a*x, a*b*(torch.exp(x) - 1.) + a*c*x)

class LSELU(nn.Module):
    def __init__(self, a:float, b:float, c:float, requires_grad:bool = False)->None:
        super().__init__()
        self.a = nn.Parameter(torch.Tensor([a]), requires_grad = requires_grad)
        self.b = nn.Parameter(torch.Tensor([b]), requires_grad = requires_grad)
        self.c = nn.Parameter(torch.Tensor([c]), requires_grad = requires_grad)

    def forward(self, x:torch.Tensor)->torch.Tensor:
        return lselu(x, self.a, self.b, self.c)

def serlu(x:torch.Tensor, a:float|torch.Tensor, b:float|torch.Tensor)->torch.Tensor:
    return torch.where(x > 0., a*x, a*b*torch.exp(x))

class SERLU(nn.Module):
    def __init__(self, a:float, b:float, requires_grad:bool = False)->None:
        super().__init__()
        self.a = nn.Parameter(torch.Tensor([a]), requires_grad = requires_grad)
        self.b = nn.Parameter(torch.Tensor([b]), requires_grad = requires_grad)

    def forward(self, x:torch.Tensor)->torch.Tensor:
        return serlu(x, self.a, self.b)

def sselu(x:torch.Tensor, a:float|torch.Tensor, b:float|torch.Tensor, c:float|torch.Tensor)->torch.Tensor:
    return torch.where(x > 0., a*x, a*b*(torch.exp(c*x) - 1.))

class sSELU(nn.Module):
    def __init__(self, a:float, b:float, c:float, requires_grad:bool = False)->None:
        super().__init__()
        self.a = nn.Parameter(torch.Tensor([a]), requires_grad = requires_grad)
        self.b = nn.Parameter(torch.Tensor([b]), requires_grad = requires_grad)
        self.c = nn.Parameter(torch.Tensor([c]), requires_grad = requires_grad)

    def forward(self, x:torch.Tensor)->torch.Tensor:
        return sselu(x, self.a, self.b, self.c)

def elish(x:torch.Tensor)->torch.Tensor:
    return torch.where(x > 0., x/(1. + torch.exp(-x)), (torch.exp(x) - 1.)/(1. + torch.exp(-x)))

class ELiSH(nn.Module):
    def forward(self, x:torch.Tensor)->torch.Tensor:
        return elish(x)