import sys
import torch

from torch.nn import functional as F
from math import log, pi, exp
from flow.utils import proc_problematic_samples

########################################################################################################################
########################################################################################################################

def gaussian_log_p(x, mean, log_sd):
    return -0.5 * log(2 * pi) - log_sd - 0.5 * (x - mean) ** 2 / torch.exp(2 * log_sd)


def gaussian_sample(eps, mean, log_sd):
    return mean + torch.exp(log_sd) * eps

class Model(torch.nn.Module):

    def __init__(self,sqfactor,nblocks,nflows,ncha,semb=128,block_per_split=3):
        super(Model,self).__init__()

        self.block_per_split = block_per_split
        self.sqfactor = sqfactor
        self.nblocks = nblocks

        nsq=10
        self.blocks=torch.nn.ModuleList()
        for b in range(nblocks):
            split = False if (b + 1) % self.block_per_split or b == nblocks - 1 else True
            self.blocks.append(Block(sqfactor,nflows,nsq,ncha,semb,block_number=b,split=split))
            if not split:
                nsq *= 2
        self.final_nsq=nsq

        self.squeeze=Squeezer(factor=sqfactor)

    def forward(self,h,emb=None):
        B, _, T = h.size()

        # Prepare
        averaging_emb = None
        if emb != None:
            averaging_emb = self.averaging_adjacent_embedding(emb)

        # Run blocks & accumulate log-det
        log_det, log_p_sum = 0, 0
        out = h
        z_list = []
        for block in self.blocks:
            out,ldet,log_p,z=block.forward(out,averaging_emb)
            if z != None: z_list.append(z)
            log_det+=ldet

            if log_p is not None:
                log_p_sum = log_p_sum + log_p
        log_p_sum += 0.5 * (- log(2.0 * pi) - out.pow(2)).sum()

        log_det = log_det / (B * T)
        log_p_sum = log_p_sum / (B * T)
        return out,log_p_sum,log_det,z_list

    def reverse(self,z,emb=None,z_list=[]):
        # Prepare
        sbatch = z.size(0)

        averaging_emb = None
        if emb != None:
            averaging_emb = self.averaging_adjacent_embedding(emb)
        
        x = z
        if len(z_list) <= 0:
            for i in range(self.nblocks):
                x = self.squeeze.forward(x)
                if not ((i + 1) % self.block_per_split or i == self.nblocks - 1):
                    x, z = x.chunk(2, 1)
                    z_list.append(z)
        # Run blocks
        for i, block in enumerate(self.blocks[::-1]):
            index = self.nblocks - i
            if not (index % self.block_per_split or index == self.nblocks):
                x = block.reverse(x, averaging_emb, z_list[index // self.block_per_split - 1])
            else:
                x = block.reverse(x, averaging_emb)
        # Back to original dim
        h=x.view(sbatch, -1)
        # Postproc
        h=proc_problematic_samples(h)
        return h

    def averaging_adjacent_embedding(self, emb):
        adjacent_num = self.sqfactor
        adjacent_emb = []
        adjacent_emb.append(emb)
        for b in range(len(self.blocks) - 1):
            emb = emb.view(emb.size(0), -1, adjacent_num).mean(axis=2).view(emb.size(0), emb.size(1), -1)
            adjacent_emb.append(emb)
        
        return adjacent_emb

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

    def __init__(self,sqfactor,nflows,nsq,ncha,semb,block_number,split=False):
        super(Block,self).__init__()

        self.block_number = block_number
        self.split = split
        squeeze_dim = nsq * 2
        
        self.squeeze=Squeezer(factor=sqfactor)

        self.flows=torch.nn.ModuleList()
        for _ in range(nflows):
            self.flows.append(Flow(squeeze_dim,ncha,semb))
        self.actnorm = ActNorm(squeeze_dim)

        if self.split:
            self.prior = CouplingNet(squeeze_dim // 2, squeeze_dim, filter_size=256)

    def forward(self,h,emb):
        # Squeeze
        out=self.squeeze.forward(h)
        # Run flows & accumulate log-det
        log_det=0
        for flow in self.flows:
            embb = None
            if emb != None:
                embb = emb[self.block_number]
            out,ldet=flow.forward(out, embb)
            log_det+=ldet
        out, ldet = self.actnorm(out)
        log_det+=ldet

        z = None
        log_p = 0
        if self.split:
            out, z = out.chunk(2, 1)
            mean, log_sd = self.prior(out).chunk(2, 1)
            log_p = gaussian_log_p(z, mean, log_sd).sum()

        return out,log_det,log_p,z

    def reverse(self,output, emb, eps=None):
        if self.split:
            mean, log_sd = self.prior(output).chunk(2, 1)
            z_new = gaussian_sample(eps, mean, log_sd)

            input = torch.cat([output, z_new], 1)
        else:
            input = output

        input = self.actnorm.reverse(input)
        # Run flows
        for flow in self.flows[::-1]:
            embb = None
            if emb != None:
                embb = emb[self.block_number]
            input=flow.reverse(input, embb)
        
        # Unsqueeze
        input=self.squeeze.reverse(input)
        return input

########################################################################################################################

class Flow(torch.nn.Module):

    def __init__(self,nsq,ncha,semb):
        super(Flow,self).__init__()

        self.norm=ActNorm(nsq)
        self.mixer=Invertible1x1Conv(nsq)
        self.coupling=AffineCoupling(nsq,semb,ncha,affine=True)

        return

    def forward(self,h,emb):
        h,logdet=self.norm.forward(h)
        h,ld2=self.mixer.forward(h)
        h,ld3=self.coupling.forward(h,emb)

        logdet = logdet + ld2
        if ld3 is not None:
            logdet = logdet + ld3
        return h,logdet

    def reverse(self,h,emb):
        h=self.coupling.reverse(h,emb)
        h=self.mixer.forward(h, reverse=True)
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

from torch.autograd import Variable

class Invertible1x1Conv(torch.nn.Module):
    """
    The layer outputs both the convolution, and the log determinant
    of its weight matrix.  If reverse=True it does convolution with
    inverse
    """
    def __init__(self, c):
        super(Invertible1x1Conv, self).__init__()
        self.conv = torch.nn.Conv1d(c, c, kernel_size=1, stride=1, padding=0,
                                    bias=False)

        # Sample a random orthonormal matrix to initialize weights
        W = torch.linalg.qr(torch.FloatTensor(c, c).normal_())[0]

        # Ensure determinant is 1.0 not -1.0
        if torch.det(W) < 0:
            W[:,0] = -1*W[:,0]
        W = W.view(c, c, 1)
        self.conv.weight.data = W

    def forward(self, z, reverse=False):
        # shape
        batch_size, group_size, n_of_groups = z.size()

        W = self.conv.weight.squeeze()

        if reverse:
            if not hasattr(self, 'W_inverse'):
                # Reverse computation
                W_inverse = W.float().inverse()
                W_inverse = Variable(W_inverse[..., None])
                if z.type() == 'torch.cuda.HalfTensor':
                    W_inverse = W_inverse.half()
                self.W_inverse = W_inverse
            z = F.conv1d(z, self.W_inverse, bias=None, stride=1, padding=0)
            return z
        else:
            # Forward computation
            log_det_W = batch_size * n_of_groups * torch.logdet(W)
            z = self.conv(z)
            return z, log_det_W
########################################################################################################################

logabs = lambda x: torch.log(torch.abs(x))

class ActNorm(torch.nn.Module):
    def __init__(self, in_channel, logdet=True):
        super().__init__()

        self.loc = torch.nn.Parameter(torch.zeros(1, in_channel, 1))
        self.scale = torch.nn.Parameter(torch.ones(1, in_channel, 1))

        # self.initialized = pretrained
        self.logdet = logdet

    # def initialize(self, x):
    #     with torch.no_grad():
    #         flatten = x.permute(1, 0, 2).contiguous().view(x.shape[1], -1)
    #         mean = (
    #             flatten.mean(1)
    #             .unsqueeze(1)
    #             .unsqueeze(2)
    #             .permute(1, 0, 2)
    #         )
    #         std = (
    #             flatten.std(1)
    #             .unsqueeze(1)
    #             .unsqueeze(2)
    #             .permute(1, 0, 2)
    #         )

    #         self.loc.data.copy_(-mean)
    #         self.scale.data.copy_(1 / (std + 1e-6))

    def forward(self, x):
        B, _, T = x.size()

        # if not self.initialized:
        #     self.initialize(x)
        #     self.initialized = True

        log_abs = logabs(self.scale)

        logdet = torch.sum(log_abs) * B * T

        if self.logdet:
            return self.scale * (x + self.loc), logdet

        else:
            return self.scale * (x + self.loc)

    def reverse(self, output):
        return output / self.scale - self.loc

########################################################################################################################

class ZeroConv1d(torch.nn.Module):
    def __init__(self, in_channel, out_channel, padding=0):
        super().__init__()

        self.conv = torch.nn.Conv1d(in_channel, out_channel, 3, padding=padding)
        self.conv.weight.data.zero_()
        self.conv.bias.data.zero_()
        self.scale = torch.nn.Parameter(torch.zeros(1, out_channel, 1))

    def forward(self, x):
        out = self.conv(x)
        out = out * torch.exp(self.scale * 3)
        return out

class AffineCoupling(torch.nn.Module):
    def __init__(self, in_channel, semb, filter_size=256, affine=True):
        super().__init__()

        self.affine = affine
        self.net=CouplingNet(in_channel // 2, in_channel if affine else in_channel // 2, filter_size, semb)

    def forward(self, x, emb=None):
        in_a, in_b = x.chunk(2, 1)

        if self.affine:
            log_s, t = self.net(in_a, emb).chunk(2, 1)

            out_b = (in_b - t) * torch.exp(-log_s)
            logdet = torch.sum(-log_s)
        else:
            net_out = self.net(in_a, emb)
            out_b = in_b + net_out
            logdet = None
        return torch.cat([in_a, out_b], 1), logdet

    def reverse(self, output, emb=None):
        out_a, out_b = output.chunk(2, 1)

        if self.affine:
            log_s, t = self.net(out_a, emb).chunk(2, 1)
            in_b = out_b * torch.exp(log_s) + t
        else:
            net_out = self.net(out_a, emb)
            in_b = out_b - net_out

        return torch.cat([out_a, in_b], 1)

class CouplingNet(torch.nn.Module):

    def __init__(self, in_channel, out_channel, filter_size=256, semb=None):
        super(CouplingNet,self).__init__()
        self.in_conv = torch.nn.Conv1d(in_channels=in_channel,
                                        out_channels=filter_size,
                                        kernel_size=3, padding=1)
        if semb != None:
            self.in_cond_conv = torch.nn.Conv1d(in_channels=semb,
                                            out_channels=in_channel,
                                            kernel_size=3, padding=1)

        self.mid_conv = torch.nn.Conv1d(in_channels=filter_size,
                               out_channels=filter_size,
                               kernel_size=1)
        if semb != None:
            self.mid_cond_conv = torch.nn.Conv1d(in_channels=semb,
                                    out_channels=filter_size,
                                    kernel_size=1)

        self.out_conv = ZeroConv1d(in_channel=filter_size,
                                    out_channel=out_channel, padding=1)

    def forward(self, h, emb=None):
        embb = 0
        if emb != None:
            embb = self.in_cond_conv(emb)
        h = self.in_conv(h + embb)
        h = F.relu(h, inplace=True)

        embb = 0
        if emb != None:
            embb = self.mid_cond_conv(emb)
        h = self.mid_conv(h + embb)
        h = F.relu(h, inplace=True)
        
        h = self.out_conv(h)
        return h

########################################################################################################################
########################################################################################################################
