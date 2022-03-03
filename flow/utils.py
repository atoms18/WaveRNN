
import torch
import math

def proc_problematic_samples(x,soft=True):
    x[torch.isnan(x)]=0
    if soft:
        x=softclamp(x)
    else:
        x=torch.clamp(x,-1,1)
    return x

def softclamp(x,mx=1,margin=0.03,alpha=0.7,clipval=100):
    x=torch.clamp(x,-clipval,clipval)
    xabs=x.abs()
    rmargin=mx-margin
    mask=(xabs<rmargin).float()
    x=mask*x+(1-mask)*torch.sign(x)*((1-torch.exp(-alpha*(xabs-rmargin)/margin))*margin+rmargin)
    return x