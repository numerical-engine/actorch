import torch
import torch.nn as nn
import torch.nn.functional as F

def vsigmoid(x:torch.Tensor, a:float|torch.Tensor, b:float|torch.Tensor, c:float|torch.Tensor)->torch.Tensor:
    return a*F.sigmoid(b*x) - c

class vSigmoid(nn.Module):
    def __init__(self, a:float = 1.0, b:float = 1.0, c:float = 0.0, requires_grad:bool = False)->None:
        super().__init__()
        self.a = nn.Parameter(torch.Tensor([a]), requires_grad = requires_grad)
        self.b = nn.Parameter(torch.Tensor([b]), requires_grad = requires_grad)
        self.c = nn.Parameter(torch.Tensor([c]), requires_grad = requires_grad)
    
    def forward(self, x:torch.Tensor)->torch.Tensor:
        return vsigmoid(x, self.a, self.b, self.c)

def siglin(x:torch.Tensor, a:float|torch.Tensor)->torch.Tensor:
    return F.sigmoid(x) + a*x

class SigLin(nn.Module):
    def __init__(self, a:float, requires_grad:bool = False)->None:
        super().__init__()
        self.a = nn.Parameter(torch.Tensor([a]), requires_grad = requires_grad)
    
    def forward(self, x:torch.Tensor)->torch.Tensor:
        return siglin(x, self.a)

def srs(x:torch.Tensor, a:float|torch.Tensor, b:float|torch.Tensor)->torch.Tensor:
    return x/((x/a) + torch.exp(-x/b))

class SRS(nn.Module):
    def __init__(self, a:float = 2.0, b:float = 3.0, requires_grad:bool = False)->None:
        super().__init__()
        self.a = nn.Parameter(torch.Tensor([a]), requires_grad = requires_grad)
        self.b = nn.Parameter(torch.Tensor([b]), requires_grad = requires_grad)
    
    def forward(self, x:torch.Tensor)->torch.Tensor:
        return srs(x, self.a, self.b)
    
def softclip(x:torch.Tensor, a:float|torch.Tensor)->torch.Tensor:
    return torch.log((1. + torch.exp(a*x))/(1. + torch.exp(a*(x-1.))))/a

class SoftClip(nn.Module):
    def __init__(self, a:float, requires_grad:bool = False)->None:
        super().__init__()
        self.a = nn.Parameter(torch.Tensor([a]), requires_grad = requires_grad)
    
    def forward(self, x:torch.Tensor)->torch.Tensor:
        return softclip(x, self.a)

def softsign(x:torch.Tensor)->torch.Tensor:
    return x/(1. + torch.abs(x))

class SoftSign(nn.Module):
    def forward(self, x:torch.Tensor)->torch.Tensor:
        return softsign(x)

def elliott(x:torch.Tensor)->torch.Tensor:
    return 0.5*x/(1. + torch.abs(x)) + 0.5

class Elliott(nn.Module):
    def forward(self, x:torch.Tensor)->torch.Tensor:
        return elliott(x)

def sigmoidgumbel(x:torch.Tensor)->torch.Tensor:
    return torch.exp(-torch.exp(-x))/(1. + torch.exp(-x))

class SigmoidGumbel(nn.Module):
    def forward(self, x:torch.Tensor)->torch.Tensor:
        return sigmoidgumbel(x)

def sigmoidnew(x:torch.Tensor)->torch.Tensor:
    return (torch.exp(x) - torch.exp(-x))/torch.sqrt(2.*torch.exp(2.*x) + torch.exp(-2.*x))

class SigmoidNew(nn.Module):
    def forward(self, x:torch.Tensor)->torch.Tensor:
        return sigmoidnew(x)

def loglog(x:torch.Tensor)->torch.Tensor:
    return torch.exp(-torch.exp(-x))

class LogLog(nn.Module):
    def forward(self, x:torch.Tensor)->torch.Tensor:
        return loglog(x)

def cloglog(x:torch.Tensor)->torch.Tensor:
    return 1. - loglog(x)

class cLogLog(nn.Module):
    def forward(self, x:torch.Tensor)->torch.Tensor:
        return cloglog(x)

def tanhsig(x:torch.Tensor)->torch.Tensor:
    return (x + F.tanh(x))*F.sigmoid(x)

class TanhSig(nn.Module):
    def forward(self, x:torch.Tensor)->torch.Tensor:
        return tanhsig(x)

def rootsig(x:torch.Tensor, a:float|torch.Tensor)->torch.Tensor:
    return a*x/(1. + torch.sqrt(1. + (a*x)**2))

class RootSig(nn.Module):
    def __init__(self, a:float, requires_grad:bool = False)->None:
        super().__init__()
        self.a = nn.Parameter(torch.Tensor([a]), requires_grad = requires_grad)
    
    def forward(self, x:torch.Tensor)->torch.Tensor:
        return rootsig(x, self.a)

def sgelu(x:torch.Tensor, a:float|torch.Tensor)->torch.Tensor:
    return a*x*torch.erf(x/math.sqrt(2.))

class SGELU(nn.Module):
    def __init__(self, a:float, requires_grad:bool = False)->None:
        super().__init__()
        self.a = nn.Parameter(torch.Tensor([a]), requires_grad = requires_grad)
    
    def forward(self, x:torch.Tensor)->torch.Tensor:
        return sgelu(x, self.a)

def colu(x:torch.Tensor)->torch.Tensor:
    return x/(1. - x*torch.exp(-(x + torch.exp(x))))

class CoLU(nn.Module):
    def forward(self, x:torch.Tensor)->torch.Tensor:
        return colu(x)

def generalizedswish(x:torch.Tensor)->torch.Tensor:
    return x*F.sigmoid(torch.exp(-x))

class GeneralizedSwish(nn.Module):
    def forward(self, x:torch.Tensor)->torch.Tensor:
        return generalizedswish(x)

def expswish(x:torch.Tensor)->torch.Tensor:
    return torch.exp(-x)*F.sigmoid(x)

class ExpSwish(nn.Module):
    def forward(self, x:torch.Tensor)->torch.Tensor:
        return expswish(x)

def gish(x:torch.Tensor)->torch.Tensor:
    return x*torch.log(2. - torch.exp(-torch.exp(x)))

class Gish(nn.Module):
    def forward(self, x:torch.Tensor)->torch.Tensor:
        return gish(x)

def logish(x:torch.Tensor)->torch.Tensor:
    return x*torch.log(1. + F.sigmoid(x))

class Logish(nn.Module):
    def forward(self, x:torch.Tensor)->torch.Tensor:
        return logish(x)

def loglogish(x:torch.Tensor)->torch.Tensor:
    return x*(1. - torch.exp(-torch.exp(x)))

class LogLogish(nn.Module):
    def forward(self, x:torch.Tensor)->torch.Tensor:
        return loglogish(x)

def expexpish(x:torch.Tensor)->torch.Tensor:
    return x*torch.exp(-torch.exp(-x))

class ExpExpish(nn.Module):
    def forward(self, x:torch.Tensor)->torch.Tensor:
        return expexpish(x)

def phish(x:torch.Tensor)->torch.Tensor:
    return x*F.tanh(F.gelu(x))

class Phish(nn.Module):
    def forward(self, x:torch.Tensor)->torch.Tensor:
        return phish(x)

def tsrelu(x:torch.Tensor)->torch.Tensor:
    return x*F.tanh(F.sigmoid(x))

class TSReLU(nn.Module):
    def forward(self, x:torch.Tensor)->torch.Tensor:
        return tsrelu(x)

def dsilu(x:torch.Tensor)->torch.Tensor:
    return F.sigmoid(x)*(1. + x*(1. - F.sigmoid(x)))

class dSiLU(nn.Module):
    def forward(self, x:torch.Tensor)->torch.Tensor:
        return dsilu(x)

def doublesilu(x:torch.Tensor)->torch.Tensor:
    return x / (1. + torch.exp(-x/(1. + torch.exp(-x))))

class DoubleSiLU(nn.Module):
    def forward(self, x:torch.Tensor)->torch.Tensor:
        return doublesilu(x)

def msilu(x:torch.Tensor)->torch.Tensor:
    return x*F.sigmoid(x) + torch.exp(-x**2 - 1.0)/4.0

class MSiLU(nn.Module):
    def forward(self, x:torch.Tensor)->torch.Tensor:
        return msilu(x)

def smish(x:torch.Tensor, a:float|torch.Tensor, b:torch.Tensor)->torch.Tensor:
    return a*x*F.tanh(torch.log(1. + F.sigmoid(b*x)))

class Smish(nn.Module):
    def __init__(self, a:float = 1., b:float = 1., requires_grad:bool = False)->None:
        super().__init__()
        self.a = nn.Parameter(torch.Tensor([a]), requires_grad = requires_grad)
        self.b = nn.Parameter(torch.Tensor([b]), requires_grad = requires_grad)

    def forward(self, x:torch.Tensor)->torch.Tensor:
        return smish(x, self.a, self.b)

def tanhexp(x:torch.Tensor)->torch.Tensor:
    return x*F.tanh(torch.exp(x))

class TanhExp(nn.Module):
    def forward(self, x:torch.Tensor)->torch.Tensor:
        return tanhexp(x)

def serf(x:torch.Tensor)->torch.Tensor:
    return x*torch.erf(torch.log(1. + torch.exp(x)))

class Serf(nn.Module):
    def forward(self, x:torch.Tensor)->torch.Tensor:
        return serf(x)

def eanaf(x:torch.Tensor)->torch.Tensor:
    return x*torch.exp(x)/(torch.exp(x) + 2.)

class EANAF(nn.Module):
    def forward(self, x:torch.Tensor)->torch.Tensor:
        return eanaf(x)

def sigsig(x:torch.Tensor)->torch.Tensor:
    return x*torch.sin(0.5*math.pi*F.sigmoid(x))

class SigSig(nn.Module):
    def forward(self, x:torch.Tensor)->torch.Tensor:
        return sigsig(x)