## convolutions for Token_embedding
import torch
import torch.nn as nn  
from torch import Tensor
import torch.nn.functional as F
import torch.nn as nn
from typing import List, Tuple,Callable,Optional
import numpy as np
import matplotlib.pyplot as plt
#from torchinfo import summary

device =torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class LayerNorm1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        # We normalize over the 'feature' dimension (dim)
        self.ln = nn.LayerNorm(dim)
    def forward(self, x):
        # Input x: [Batch*Channels, Features, Time]
        # 1. Move Features to the last dimension
        x = x.transpose(1, 2) # [Batch*Channels, Time, Features]
        # 2. Normalize
        # This calculates mean/std across the 'Features' for each 'Time' step
        x = self.ln(x)
        # 3. Move back to Conv format
        return x.transpose(1, 2)

class Fp32GroupNorm(nn.GroupNorm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, input):
        output = F.group_norm(
            input.float(),
            self.num_groups,
            self.weight.float() if self.weight is not None else None,
            self.bias.float() if self.bias is not None else None,
            self.eps,
        )
        
        return output.type_as(input)
###modify the following conv_extraction module with the dilations
### pass stride and the kernel size as parameters to control the overlap
### dilation as parameter 
class ConvFeatureExtraction(nn.Module):
    def __init__(self,d_conv,conv_layers:List[Tuple[int, int, int]],dropout: float = 0.0,conv_bias: bool = False):
        super().__init__()
        ##assert mode in {"default", "layer_norm"}
        #self.layernorm1D=LayerNorm1d
        in_d = 1 ## input_channel size
        self.d_conv=d_conv
        
        def block(n_in,n_out,k,stride,dilation,conv_bias=False,):
            def make_conv():
                conv = nn.Conv1d(n_in, n_out,k,stride=stride,dilation=dilation,padding=2,bias=conv_bias)
                nn.init.kaiming_normal_(conv.weight)
                return conv
            
            return nn.Sequential(
                    make_conv(),
                    nn.Dropout(p=dropout),
                    LayerNorm1d(n_out),
                    #Fp32GroupNorm(dim, dim, affine=True),
                    nn.GELU(),
                )
        
        self.conv_layers = nn.ModuleList()
        for i, cl in enumerate(conv_layers):
            assert len(cl) == 4, "invalid conv definition: " + str(cl)
            (dim,k,stride,dilation) = cl
            self.conv_layers.append(block(in_d,dim,k,
                                        stride,dilation,conv_bias=conv_bias))
            
            in_d = dim
        
    def forward(self, x):
        b,ch,l=x.shape
        x=x.view(b*ch,-1,l).contiguous()
        for conv in self.conv_layers:
            x = conv(x)            
        x.contiguous()
        return x.permute(0,2,1).view(b,ch,-1,self.d_conv) #shape (batch*ch,pseudo_ts,feature)

class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        self.c_in=c_in
        self.d_model=d_model
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.tokenConv = nn.Conv1d(in_channels=1, out_channels=d_model,
                                   kernel_size=5, stride=2,padding=0,dilation=2,padding_mode='circular', bias=False)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        b,c,l = x.size()
        x=x.contiguous()
        x_reshaped=x.view(b*c,-1,l)
        out = self.tokenConv(x_reshaped)
        out=out.contiguous()
        return out.view(c,-1,self.d_model)

###ts_data of seq_len=400 and c_in=10 (N,C,L)
"""ts_data=torch.randn(1,1,322)
conv_layers_1=[(64,5,3,1),(128,5,3,2),(256,3,2,2),(512,3,2,2),(1024,3,2,2)]
conv_layer_2=[(64,7,3,1),(128,5,3,1),(256,3,2,1),(512,3,2,1)]
conv_model = ConvFeatureExtraction(1024,conv_layers_1,dropout=0.1,conv_bias=True)
conv_embed = conv_model(ts_data)
#summary(conv_model,input_size=ts_data.shape)
print(conv_embed.shape)"""
###print(ts_embed[:,:3,:5])
