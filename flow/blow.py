import sys
import torch

from torch.nn import functional as F
from math import log, pi, exp
# from utils import audio

########################################################################################################################
########################################################################################################################

def gaussian_log_p(x, mean, log_sd):
    return -0.5 * log(2 * pi) - log_sd - 0.5 * (x - mean) ** 2 / torch.exp(2 * log_sd)


def gaussian_sample(eps, mean, log_sd):
    return mean + torch.exp(log_sd) * eps

class Model(torch.nn.Module):

    def __init__(self,sqfactor,nblocks,nflows,ncha,ntargets,_,semb=128):
        super(Model,self).__init__()

        nsq=10
        print('Channels/squeeze = ',end='')
        self.blocks=torch.nn.ModuleList()
        for _ in range(nblocks - 1):
            self.blocks.append(Block(sqfactor,nflows,nsq,ncha,semb))
            print('{:d}, '.format(nsq),end='')
            nsq*=sqfactor
        self.blocks.append(Block(sqfactor,nflows,nsq,ncha,semb))
        print('{:d}, '.format(nsq),end='')
        print()
        self.final_nsq=nsq

        # self.embedding=torch.nn.Embedding(ntargets,semb)

        return

    def forward(self,h):
        # Prepare
        sbatch,begin_nsq,begin_lchunk=h.size()
        # h=h.unsqueeze(1)
        # emb=self.embedding(s)
        # Run blocks & accumulate log-det
        log_det=0
        z_lists = []
        log_p_sum = 0
        for block in self.blocks:
            h,ldet,log_p=block.forward(h)
            z_lists.append(h)
            log_det+=ldet

            if log_p is not None:
                log_p_sum = log_p_sum + log_p
        # Back to original dim
        z_last=h.view(sbatch,-1)
        return z_last,log_p_sum,log_det,z_lists

    def reverse(self,z_list,reconstruct=False):
        # Prepare
        sbatch,nsq,lchunk=z_list[-1].size()
        # h=h.view(sbatch,self.final_nsq,lchunk//self.final_nsq)
        # emb=self.embedding(s)
        # Run blocks
        h = z_list[-1]
        for i, block in enumerate(self.blocks[::-1]):
            if i == 0:
                h = block.reverse(h, h, reconstruct=reconstruct)
            else:
                prev_z_list = h if len(z_list) == 1 else z_list[-(i + 1)]
                h = block.reverse(h, prev_z_list, reconstruct=reconstruct)
        # Back to original dim
        h=h.view(sbatch,-1)
        # Postproc
        # h=audio.proc_problematic_samples(h)
        return h

    def precalc_matrices(self,mode):
        if mode!='on' and mode!='off':
            print('[precalc_matrices() needs either on or off]')
            sys.exit()
        if mode=='off':
            for i in range(len(self.blocks)):
                for j in range(len(self.blocks[i].flows)):
                    self.blocks[i].flows[j].mixer.weight=None
                    self.blocks[i].flows[j].mixer.invweight=None
        else:
            for i in range(len(self.blocks)):
                for j in range(len(self.blocks[i].flows)):
                    self.blocks[i].flows[j].mixer.weight=self.blocks[i].flows[j].mixer.calc_weight()
                    self.blocks[i].flows[j].mixer.invweight=self.blocks[i].flows[j].mixer.weight.inverse()
        return

########################################################################################################################

class Block(torch.nn.Module):

    def __init__(self,sqfactor,nflows,nsq,ncha,semb):
        super(Block,self).__init__()

        self.squeeze=Squeezer(factor=sqfactor)
        self.flows=torch.nn.ModuleList()
        for _ in range(nflows):
            self.flows.append(Flow(nsq,ncha,semb))
        
        self.actnorm = ActNorm(nsq)

        self.prior = torch.nn.Conv1d(nsq, nsq * 2, 3, padding=1)
        self.prior.weight.data.zero_()
        self.prior.bias.data.zero_()
        

        return

    def forward(self,h):
        # Squeeze
        h=self.squeeze.forward(h)
        # Run flows & accumulate log-det
        log_det=0
        for flow in self.flows:
            h,ldet=flow.forward(h)
            log_det+=ldet
        
        z, ldet = self.actnorm(h)
        log_det+=ldet

        zero = torch.zeros_like(h)
        mean, log_sd = self.prior(zero).chunk(2, 1)
        log_p = gaussian_log_p(z, mean, log_sd)
        log_p = log_p.view(z.size(0), -1).sum(1)

        return z,log_det,log_p

    def reverse(self,z,eps=None,reconstruct=False):
        input = z

        if reconstruct:
            input = eps
        else:
            zero = torch.zeros_like(input)
            # zero = F.pad(zero, [1, 1, 1, 1], value=1)
            mean, log_sd = self.prior(zero).chunk(2, 1)
            z = gaussian_sample(eps, mean, log_sd)
            input = z


        input = self.actnorm.reverse(input)

        # Run flows
        for flow in self.flows[::-1]:
            input=flow.reverse(input)
        
        # Unsqueeze
        input=self.squeeze.reverse(input)
        return input

########################################################################################################################

class Flow(torch.nn.Module):

    def __init__(self,nsq,ncha,semb):
        super(Flow,self).__init__()

        self.norm=ActNorm(nsq)
        self.mixer=InvConv(nsq)
        self.coupling=AffineCoupling(nsq,ncha,affine=True)

        return

    def forward(self,h):
        logdet=0
        h,ld=self.norm.forward(h)
        logdet=logdet+ld
        h,ld=self.mixer.forward(h)
        logdet=logdet+ld
        h,ld=self.coupling.forward(h)
        logdet=logdet+ld
        return h,logdet

    def reverse(self,h):
        h=self.coupling.reverse(h)
        h=self.mixer.reverse(h)
        h=self.norm.reverse(h)
        return h

########################################################################################################################
########################################################################################################################

class Squeezer(object):
    def __init__(self,factor=2):
        self.factor=factor
        return

    def forward(self,h):
        sbatch,nsq,lchunk=h.size()
        h=h.view(sbatch,nsq,lchunk//self.factor,self.factor)
        h=h.permute(0,1,3,2).contiguous()
        h=h.view(sbatch,nsq*self.factor,lchunk//self.factor)
        return h

    def reverse(self,h):
        sbatch,nsq,lchunk=h.size()
        h=h.view(sbatch,nsq//self.factor,self.factor,lchunk)
        h=h.permute(0,1,3,2).contiguous()
        h=h.view(sbatch,nsq//self.factor,lchunk*self.factor)
        return h

########################################################################################################################

from scipy import linalg
import numpy as np

class InvConv(torch.nn.Module):

    def __init__(self,in_channel):
        super(InvConv,self).__init__()

        weight=np.random.randn(in_channel,in_channel)
        q,_=linalg.qr(weight)
        w_p,w_l,w_u=linalg.lu(q.astype(np.float32))
        w_s=np.diag(w_u)
        w_u=np.triu(w_u,1)
        u_mask=np.triu(np.ones_like(w_u),1)
        l_mask=u_mask.T

        self.register_buffer('w_p',torch.from_numpy(w_p))
        self.register_buffer('u_mask',torch.from_numpy(u_mask))
        self.register_buffer('l_mask',torch.from_numpy(l_mask))
        self.register_buffer('l_eye',torch.eye(l_mask.shape[0]))
        self.register_buffer('s_sign',torch.sign(torch.tensor(w_s)))
        self.w_l=torch.nn.Parameter(torch.from_numpy(w_l))
        self.w_s=torch.nn.Parameter(torch.log(1e-7+torch.abs(torch.tensor(w_s))))
        self.w_u=torch.nn.Parameter(torch.from_numpy(w_u))

        self.weight=None
        self.invweight=None

        return

    def calc_weight(self):
        weight=(
            self.w_p
            @ (self.w_l*self.l_mask+self.l_eye)
            @ (self.w_u*self.u_mask+torch.diag(self.s_sign*(torch.exp(self.w_s))))
        )
        return weight

    def forward(self,h):
        if self.weight is None:
            weight=self.calc_weight()
        else:
            weight=self.weight
        h=torch.nn.functional.conv1d(h,weight.unsqueeze(2))
        logdet=self.w_s.sum()*h.size(2)
        return h,logdet

    def reverse(self,h):
        if self.invweight is None:
            invweight=self.calc_weight().inverse()
        else:
            invweight=self.invweight
        h=torch.nn.functional.conv1d(h,invweight.unsqueeze(2))
        return h

########################################################################################################################

logabs = lambda x: torch.log(torch.abs(x))

class ActNorm(torch.nn.Module):
    def __init__(self, in_channel, logdet=True):
        super().__init__()

        self.loc = torch.nn.Parameter(torch.zeros(1, in_channel, 1))
        self.scale = torch.nn.Parameter(torch.ones(1, in_channel, 1))

        # self.register_buffer("initialized", torch.tensor(0, dtype=torch.uint8))
        self.logdet = logdet

    # def initialize(self, input):
    #     with torch.no_grad():
    #         flatten = input.permute(1, 0, 2, 3).contiguous().view(input.shape[1], -1)
    #         mean = (
    #             flatten.mean(1)
    #             .unsqueeze(1)
    #             .unsqueeze(2)
    #             .unsqueeze(3)
    #             .permute(1, 0, 2, 3)
    #         )
    #         std = (
    #             flatten.std(1)
    #             .unsqueeze(1)
    #             .unsqueeze(2)
    #             .unsqueeze(3)
    #             .permute(1, 0, 2, 3)
    #         )

    #         self.loc.data.copy_(-mean)
    #         self.scale.data.copy_(1 / (std + 1e-6))

    def forward(self, input):
        _, _, width = input.shape

        # if self.initialized.item() == 0:
        #     self.initialize(input)
        #     self.initialized.fill_(1)

        log_abs = logabs(self.scale)

        logdet = width * torch.sum(log_abs)

        if self.logdet:
            return self.scale * (input + self.loc), logdet

        else:
            return self.scale * (input + self.loc)

    def reverse(self, output):
        return output / self.scale - self.loc

########################################################################################################################

class AffineCoupling(torch.nn.Module):
    def __init__(self, in_channel, filter_size=256, affine=True):
        super().__init__()

        self.affine = affine

        self.net = torch.nn.Sequential(
            torch.nn.Conv1d(in_channel // 2, filter_size, 3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv1d(filter_size, filter_size, 1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv1d(filter_size, in_channel if self.affine else in_channel // 2, 3, padding=1),
        )

        self.net[0].weight.data.normal_(0, 0.05)
        self.net[0].bias.data.zero_()

        self.net[2].weight.data.normal_(0, 0.05)
        self.net[2].bias.data.zero_()

        self.net[-1].weight.data.zero_()
        self.net[-1].bias.data.zero_()

    def forward(self, input):
        in_a, in_b = input.chunk(2, 1)

        if self.affine:
            log_s, t = self.net(in_a).chunk(2, 1)
            # s = torch.exp(log_s)
            s = torch.sigmoid(log_s + 2)+1e-7
            # out_a = s * in_a + t
            out_b = (in_b + t) * s

            logdet = torch.sum(torch.log(s).view(input.shape[0], -1), 1)

        else:
            net_out = self.net(in_a)
            out_b = in_b + net_out
            logdet = None

        return torch.cat([in_a, out_b], 1), logdet

    def reverse(self, output):
        out_a, out_b = output.chunk(2, 1)

        if self.affine:
            log_s, t = self.net(out_a).chunk(2, 1)
            # s = torch.exp(log_s)
            s = torch.sigmoid(log_s + 2)+1e-7
            # in_a = (out_a - t) / s
            in_b = out_b / s - t

        else:
            net_out = self.net(out_a)
            in_b = out_b - net_out

        return torch.cat([out_a, in_b], 1)

# class AffineCoupling(torch.nn.Module):

#     def __init__(self,nsq,ncha,semb):
#         super(AffineCoupling,self).__init__()
#         self.net=CouplingNet(nsq//2,ncha,semb)
#         return

#     def forward(self,h):
#         h1,h2=torch.chunk(h,2,dim=1)
#         s,m=self.net.forward(h1)
#         h2=s*(h2+m)
#         h=torch.cat([h1,h2],1)
#         logdet=s.log().sum(2).sum(1)
#         return h,logdet

#     def reverse(self,h,emb):
#         h1,h2=torch.chunk(h,2,dim=1)
#         s,m=self.net.forward(h1,emb)
#         h2=h2/s-m
#         h=torch.cat([h1,h2],1)
#         return h


# class CouplingNet(torch.nn.Module):

#     def __init__(self,nsq,ncha,semb,kw=3):
#         super(CouplingNet,self).__init__()
#         assert kw%2==1
#         # assert ncha%nsq==0
#         self.ncha=ncha
#         self.kw=kw
#         # self.adapt_w=torch.nn.Linear(semb,ncha*kw)
#         # self.adapt_b=torch.nn.Linear(semb,ncha)
#         self.net=torch.nn.Sequential(
#             torch.nn.Conv1d(nsq,ncha,3),
#             torch.nn.ReLU(inplace=True),
#             torch.nn.Conv1d(ncha,ncha,1),
#             torch.nn.ReLU(inplace=True),
#             torch.nn.Conv1d(ncha,2*nsq,kw,padding=kw//2),
#         )
#         self.net[-1].weight.data.zero_()
#         self.net[-1].bias.data.zero_()
#         return

#     def forward(self,h):
#         sbatch,nsq,lchunk=h.size()
#         h=h.contiguous()
#         """
#         # Slower version
#         ws=list(self.adapt_w(emb).view(sbatch,self.ncha,1,self.kw))
#         bs=list(self.adapt_b(emb))
#         hs=list(torch.chunk(h,sbatch,dim=0))
#         out=[]
#         for hi,wi,bi in zip(hs,ws,bs):
#             out.append(torch.nn.functional.conv1d(hi,wi,bias=bi,padding=self.kw//2,groups=nsq))
#         h=torch.cat(out,dim=0)
#         """
#         # Faster version fully using group convolution
#         # w=self.adapt_w(emb).view(-1,1,self.kw)
#         # b=self.adapt_b(emb).view(-1)
#         # h=torch.nn.functional.conv1d(h.view(1,-1,lchunk),w,bias=b,padding=self.kw//2,groups=sbatch*nsq).view(sbatch,self.ncha,lchunk)
#         #"""
#         h=self.net.forward(h)
#         s,m=torch.chunk(h,2,dim=1)
#         s=torch.sigmoid(s+2)+1e-7
#         return s,m

########################################################################################################################
########################################################################################################################
