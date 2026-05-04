import torch 
import torch.nn as nn  
from torch import Tensor
import torch.nn.functional as F
import torch.nn as nn
from typing import List, Tuple,Callable,Optional
import numpy as np
import matplotlib.pyplot as plt
##import seaborn as sns
##from conv_module import ConvFeatureExtractionModel
import math
from modules.embed_conv import ConvFeatureExtraction
#from torchinfo import summary

###device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
### Resampler to latent_dim
class perceiver_resampler(nn.Module):
    def __init__(self,max_ch,lat_dim,d_embed,n_heads):
        super(perceiver_resampler,self).__init__()
        self.max_ch =max_ch
        self.lat_dim=lat_dim
        self.n_heads=n_heads
        
        self.d_q = d_embed//n_heads
        self.d_k = d_embed//n_heads
        self.d_v= d_embed//n_heads
        
        ### initialize the latent queries
        self.latent_q=nn.Parameter(torch.empty(self.max_ch*self.lat_dim,self.d_q*n_heads))
        ##self.intialize_weights_()
        #nn.init.xavier_uniform_(self.latent_q)
        ## transformation of k,v 
        self.w_k=nn.Linear(d_embed,self.d_k*n_heads)
        self.w_v=nn.Linear(d_embed,self.d_v*n_heads)
        self.scale = nn.Parameter(torch.tensor(self.n_heads ** -0.5), requires_grad=False)
        
    def intialize_weights_(self,trained_tensor=None):
        with torch.no_grad():
           if trained_tensor is not None:
               expert_seed =trained_tensor[:self.lat_dim,:]
               expanded = expert_seed.repeat(self.max_ch, 1)
               self.latent_q.copy_(expanded)
           else:
               nn.init.xavier_uniform_(self.latent_q)
        
        ###load the weights from the ts_encoder 's latent query
        ### expand the [:5,:]
        pass
        ##
    def forward(self,x,ch_mask=None):
        #print(f'x_shape_start_resampler:{x.shape}')
        b,T,d_conv = x.shape
        x_reshaped = x.reshape(-1,T,d_conv)
        print(x_reshaped.shape)
        ##self.latent_q[]
        latent_q=self.latent_q.view(-1,self.n_heads,self.max_ch*self.lat_dim,self.d_q)
        k=self.w_k(x_reshaped).view(-1,T,self.n_heads,self.d_k).permute(0,2,3,1) ###after transpose
        v=self.w_v(x_reshaped).view(-1,T,self.n_heads,self.d_v).permute(0,2,1,3)
        attn_scores = torch.matmul(latent_q,k)*self.scale
        #print(f'attn_scores :{attn_scores.shape}')
        ###apply the ch_mask
        #ch_mask=ch_mask.unsqueeze(1).expand(self.n_heads,-1,c*n).unsqueeze(0)
        ch_mask=ch_mask.unsqueeze(1).expand(-1,self.n_heads,self.max_ch*self.lat_dim,T)
        ###print(f'ch_mask:{ch_mask.shape}')
        #print(f'ch_mask:{ch_mask[0,1,:,:]}')
        attn_scores=attn_scores.masked_fill(~(ch_mask),-1e9)
        ##print(attn_scores)
        attn_weights =F.softmax(attn_scores,dim=-1)
        #print(attn_weights)
        attn_weights=attn_weights.masked_fill(~(ch_mask),float(0.))
        v=torch.matmul(attn_weights,v)
        print(f'v_shape:{v.shape}')
        v=v.transpose(1,2).contiguous().view(b,-1,self.n_heads*self.d_v)
        #print(f'values:{v.shape}')
        ##print(f'values:{v[:,:5]}')
        ## concat the heads along the features axis
        return v      
##transpose util
class Transpose(nn.Module):
    def __init__(self, *dims,contiguous=False):
        super(Transpose, self).__init__()
        self.dims,self.contiguous= dims,contiguous
        
    def forward(self, x):
        if self.contiguous: return x.transpose(*self.dims).contiguous()
        else: return x.transpose(*self.dims)
##Either positional_embeding or ALiBi positional bias
###fixed temporal embedding
class PositionalEmbedding(nn.Module):
    def __init__(self,d_model, max_len=500):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self,x,t,ch,b):
        self.pe_shape=self.pe[:t,:].expand(b,ch,-1,-1)
        print(f'pe_shape:{self.pe_shape.shape}')
        return self.pe_shape

class channel_embedding(nn.Module):
    def __init__(self, c_in, d_model,device):
        super(channel_embedding, self).__init__()
        w = torch.zeros(c_in, d_model).float()
        w.require_grad = True
        self.device =device

        position = torch.arange(0, c_in).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        w[:, 0::2] = torch.sin(position * div_term)
        w[:, 1::2] = torch.cos(position * div_term)

        self.emb = nn.Embedding(c_in, d_model,device=self.device)
        self.emb.weight = nn.Parameter(w, requires_grad=True)

    def forward(self, x):
        return self.emb(x)

class DataEmbedding(nn.Module):
    def __init__(self,d_conv=1024,conv_layers=None,max_ch=21,device=None):
        super(DataEmbedding,self).__init__()
        self.d_conv=d_conv
        self.conv_layers=conv_layers
        self.max_ch=max_ch
        self.device=device
        
        self.conv_features=ConvFeatureExtraction(self.d_conv,self.conv_layers,dropout=0.1,conv_bias=True)
        self.temporal_pos=PositionalEmbedding(self.d_conv,max_len=500)
        self.ch_pos=channel_embedding(self.max_ch,self.d_conv,self.device)
        
    def forward(self,x):
        x_conv = self.conv_features(x)
        #print(x_conv.shape)
        b,c_in,t,d_conv=x_conv.shape
        print(f'x_conv:{x_conv.shape}')
        #x_conv_reshaped=x_conv.reshape(-1,c_in*t,d_conv)
        ch_ids=torch.arange(c_in,device=x.device)
        ch_pos_embed=self.ch_pos(ch_ids).view(b,c_in,1,-1).expand(b,c_in,-1,-1)
        #ch_pos_embed.to(self.device)
        print(f'ch_pos:{ch_pos_embed.shape}')
        ##print(self.temporal_pos(x_conv).shape)
        x_pos = x_conv + self.temporal_pos(x_conv,t,c_in,b)+ ch_pos_embed
        #x_pos.to(self.device)
        
        return x_pos.reshape(b,c_in*t,-1)
      
class positional_embedding(nn.Module):
    def __init__(self,max_N,max_channel,patch_len,device=None):
        super(positional_embedding,self).__init__()
        self.max_N=max_N ##for number of patches
        self.max_channel=max_channel
        self.embedding=patch_len
        self.device=device
        ## positional encoding for the patch position and channel indices
        self.patch_pos=nn.Embedding(self.max_N,self.embedding,)  ###dictionary to store the positional information of the observation
        self.ch_pos = nn.Embedding(self.max_channel,self.embedding) ## dictionary to store the embeddings for channel information
    
        self.patch_indices=torch.arange(self.max_N).unsqueeze(0).to(self.device) ### [0,1,2...max_N-1]
        self.channel_indices=torch.arange(self.max_channel).unsqueeze(0).to(self.device) ##[0,1,2...max_channel-1]
    
    def forward(self,x:torch.tensor,ts_token_mask:bool,ch_mask:bool):
        ##shape of x post conv_extraction: (bs,N,c_in,d)
        bs,max_ch,max_N,d = x.shape

        actual_idx = self.patch_indices[ts_token_mask]
        actual_patch=self.channel_indices[ch_mask]
        
        indices_for_patch_pos=torch.where(self.patch_indices>=len(actual_idx),torch.tensor(self.max_N),self.patch_indices)
        indices_for_channel_idx=torch.where(self.channel_indices>=len(actual_patch),torch.tensor(self.max_channel),self.channel_indices)
        
        pos_embeds_tokens=self.patch_pos(indices_for_patch_pos) ## (max_N,d)
        pos_embeds_ch=self.ch_pos(indices_for_channel_idx) ## (max_ch,d)
        pos_embeds_tokens.contiguous()
        pos_embeds_ch.contiguous()
        new_pos_tokens = pos_embeds_tokens.expand(1,self.max_channel,pos_embeds_tokens.shape[1],pos_embeds_tokens.shape[2]) ## (1,max_ch,max_N,d) broadcasted to match the incoming tensor x
        
        ch_pos_reshaped = pos_embeds_ch.view(max_ch,-1,self.embedding)
        copied_pos_idx=(ch_pos_reshaped.expand(1,ch_pos_reshaped.shape[0],10,ch_pos_reshaped.shape[2]))
        x=x+new_pos_tokens+copied_pos_idx
        
        return x

class AlibiBlock(nn.Module):
    def __init__(self,n_heads,max_n):
        super(AlibiBlock,self).__init__()
        self.n_heads=n_heads
        self.max_n=max_n
        slope_list=self._get_slopes(self.n_heads)
        slopes_tensor = torch.tensor(slope_list).view(self.n_heads, 1, 1)
        self.register_buffer("slopes", slopes_tensor)
        ##print(self.slopes)
        pos = torch.arange(self.max_n)
        dist = -torch.abs(pos.unsqueeze(1) - pos.unsqueeze(0)) ### build master tensor for relative distance (i-j)
        self.register_buffer("master_dist_block", dist)
        ##print(self.master_dist_block)
        
    def _get_slopes(self,n):
        def get_slopes_power_of_2(n):
            start = (2**(-2**-(math.log2(n)-3)))
            ratio = start
            return [start*ratio**i for i in range(n)]

        if math.log2(n).is_integer():
            return get_slopes_power_of_2(n)                   #In the paper, we only train models that have 2^a heads for some a. This function has
        else:                                                 #some good properties that only occur when the input is a power of 2. To maintain that even            #when the number of heads is not a power of 2, we use this workaround.
            closest_power_of_2 = 2**math.floor(math.log2(n)) 
            return get_slopes_power_of_2(closest_power_of_2)+self._get_slopes(2*closest_power_of_2)[0::2][:n-closest_power_of_2]
        
    def forward(self,attn_score,N,ch):
        ##logic to multiply the slop
        current_block = self.master_dist_block[:N,:N]
        full_dist_block=current_block.repeat((ch,ch))
        alibi_bias = self.slopes * full_dist_block
        ##print(f'alibi_bias{alibi_bias[:,:4,:4]}')
        
        return attn_score + alibi_bias

class ScaledDotProductAttention(nn.Module):
    ###ALibi based relative positional bias implemented
    def __init__(self,d_model,n_heads,attn_dropout=0.1,res_attention=False,lsa=False):
        super(ScaledDotProductAttention, self).__init__()
        self.attn_dropout=nn.Dropout(attn_dropout)
        self.res_attention = res_attention  ## boolean to use attention
        self.n_heads=n_heads
        self.head_dim = d_model // n_heads
        self.lsa=lsa
        self.alibi_pos=AlibiBlock(n_heads,20)
        self.scale = nn.Parameter(torch.tensor(self.head_dim ** -0.5), requires_grad=self.lsa)
        ## boolean to use learnable scale

    def forward(self, q:Tensor, k:Tensor, v:Tensor,ts_mask:bool=None,actual_N=None,actual_ch=None):
        attn_scores =torch.matmul(q,k)*self.scale
        attn_scores=self.alibi_pos(attn_scores,actual_N,actual_ch)      
       
        ##print(f'attention_logitsd :{attn_scores[0,0,:,:]}')
        if ts_mask is not None:
            attn_scores.masked_fill_(~(ts_mask.unsqueeze(0).unsqueeze(1)),float('-inf'))  ### resized the mask to the same as the attention matrix
        else:
            pass
        
        attn_weights =F.softmax(attn_scores, dim=-1)
        ##print(f'attention_weights{attn_weights.shape}')
        attn_weights= self.attn_dropout(attn_weights)   ##regularization
        output = torch.matmul(attn_weights, v) ## v_tensor shape: [bs X n_heads X q_len X d_v]
        
        if self.res_attention:
            return output,attn_weights,attn_scores ## return the out in the order output,scores
        else:
            return output,attn_weights

class MultiheadAttention(nn.Module):
    def __init__(self,d_model,n_heads,d_k=None,d_v=None,res_attention=False,
                 attn_dropout=0.1,proj_dropout=0.1,qkv_bias=True,lsa=False):
        
        super(MultiheadAttention, self).__init__()
        d_k=d_model//n_heads if d_k is None else d_k
        d_v=d_model//n_heads if d_v is None else d_v        
        self.n_heads,self.d_k,self.d_v=n_heads,d_k,d_v
        self.res_attention=res_attention   ## boolean to use attention
        
        self.W_Q=nn.Linear(d_model,self.d_k*n_heads,bias=qkv_bias)
        self.W_K=nn.Linear(d_model,self.d_k*n_heads,bias=qkv_bias)
        self.W_V=nn.Linear(d_model,self.d_v*n_heads,bias=qkv_bias)

        self.attention=ScaledDotProductAttention(d_model,n_heads,res_attention=res_attention,attn_dropout=attn_dropout,lsa=lsa)

        ##project to the output
        self.to_out= nn.Sequential(nn.Linear(n_heads*self.d_v,d_model),
                                   nn.Dropout(proj_dropout))
        
    def forward(self,Q:Tensor,K:Tensor,V:Tensor,mask=None,actual_N=None,actual_ch=None):
        
        bs=Q.size(0)
        if K is None: K=Q
        if V is None:V=Q
        
        ##linear transformation of input tensor 'x' into Q,K and V
        q_s = self.W_Q(Q).view(bs,-1,self.n_heads,self.d_k).transpose(1,2) ##  (bs,n_heads,seq_len,d_k)
        k_s = self.W_K(K).view(bs,-1,self.n_heads,self.d_k).permute(0,2,3,1) ##  (bs,n_heads,seq_len,d_k)
        v_s = self.W_V(V).view(bs,-1,self.n_heads,self.d_v).transpose(1,2) ##  (bs,n_heads,seq_len,d_v)
        
        if self.res_attention:
            output,attn_scores,attn_weights= self.attention(q_s,k_s,v_s,ts_mask=mask,actual_N=actual_N,actual_ch=actual_ch)        
        else:
            output,attn_weights=self.attention(q_s,k_s,v_s,ts_mask=mask,actual_N=actual_N,actual_ch=actual_ch)
        ## output:[bs X n_heads X q_len X d_v]
        
        ## reassemble invidual heads to get MHSA output
        output= output.transpose(1,2).contiguous().view(bs,-1,self.n_heads*self.d_v) ## (bs,seq_len,n_heads*d_v) 

        output= self.to_out(output)
        
        if self.res_attention: 
            return output,attn_scores,attn_weights
        else: 
            return output,attn_weights 

class TS_encoder_layer(nn.Module):
    def __init__(self,d_model,n_heads,max_ch=5,d_ff=256,lat_dim=5,
                 dropout=0.,bias=False,pre_norm=True):
        super(TS_encoder_layer, self).__init__()
        
        self.pre_norm = pre_norm
        self.lat_dim=lat_dim
        assert not d_model%n_heads, f'd_model ({d_model}) must be divisible by n_heads ({n_heads})'
        self.d_k = d_model // n_heads
        self.d_v = d_model // n_heads
        
        ##use Layer norm for attention sublayer pre-norm
        self.norm_attn = nn.LayerNorm(d_model)
        ## perceiver-resampler 
        self.cross_attn=perceiver_resampler(max_ch,self.lat_dim,d_model,n_heads)
        self.dropout_attn= nn.Dropout(dropout)
        ##use Layer norm for FFN sublayer pre-norm
        self.norm_ffn = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(nn.Linear(d_model,d_ff,bias=bias),
                                 nn.GELU(),
                                 nn.Dropout(dropout),
                                 nn.Linear(d_ff,d_model,bias=bias))
        ##add & norm 
        self.dropout_ffn=nn.Dropout(dropout)
        
         ## optional to normalize prior to encoder block
    def forward(self, src:Tensor,ch_mask=None) -> Tensor:
        # pre-norm
        if self.pre_norm:
            src = self.norm_attn(src)
        # Sublayer-1. Multi-Head cross-attention with latent_q : sublayer-1
        src_resampled= self.cross_attn(src,ch_mask=ch_mask)
        print(f'cross_attention:{src_resampled.shape}')
        
        if self.pre_norm:
            src_resampled = self.norm_ffn(src_resampled)
        # Sublayer-2. Position-wise Feed-Forward :sublayer-2
        ffn_src = self.ff(src_resampled)
        #print(f'out_ffn{ffn_src.shape}')
        ## Add & Norm
        src_post = src_resampled + self.dropout_ffn(ffn_src) # Add: residual connection with residual dropout
        print(f'output_resampled:{src_post.shape}')
        ##src = (src*attn_mask.to(src.dtype).unsqueeze(-1)) ## to zeros out the padded tokens after FFN
        return src_post

class TST_encoder(nn.Module):
     def __init__(self,max_ch=21,lat_dim=5,d_model=None,n_heads=2,d_ff=256,norm='BatchNorm',bias=False,
                 dropout=0.,n_layers=1,res_attention=False,pre_norm=True):

        super(TST_encoder,self).__init__()
        self.cross_attn_block=TS_encoder_layer(d_model,n_heads,max_ch=max_ch,lat_dim=lat_dim,
                                               d_ff=d_ff,dropout=dropout,bias=bias,pre_norm=pre_norm)
        """ self.layers = nn.ModuleList([
            TS_encoder_layer(max_ch=max_ch,d_model=d_model,n_heads=n_heads,lat_dim=lat_dim,d_ff=d_ff,dropout=dropout,bias=bias,pre_norm=pre_norm)
            for _ in range(n_layers)
        ])"""
        ##self.res_attention = res_attention
    
     def forward(self, src: torch.Tensor,ch_mask=None):
        output=src ### initial input tensor
        #scores=None
        output = self.cross_attn_block(output,ch_mask=ch_mask)
        #for mod in self.layers: output = mod(output,ch_mask=ch_mask)
        return output

##input_shape --> B,C*N,p
class PatchTSTEncoder(nn.Module):
    
    """ 1. conv_feature extractor to create data embedding from raw timeseries data
        2. TST_transformer for inter- intra- channel attention
    Args:
        d_conv (int): the final feature size of convolution layers
        conv_layers(list):list if tuples (dim,k,stride,dilation) 
        max_ch: maximum channel size 
        lat_dim : latent query dimension
        d_model: final projection dimension
    """
    def __init__(self,conv_layers:List,d_conv=1024,max_ch=21,n_layers=1,d_model=128,n_heads=8,d_ff=256,lat_dim=5,
                 dropout=0.1,activation='gelu',pre_norm=True,bias=None,device=None,**kwargs):
        
        super().__init__()
        self.d_model=d_model 
        self.activation=activation
        self.n_heads=n_heads
        
        self.data_embedding = DataEmbedding(d_conv=d_conv,conv_layers=conv_layers,max_ch=max_ch,device=device)
        ##Encoder
        self.encoder=TST_encoder(max_ch=max_ch,lat_dim=lat_dim,d_model=d_conv,n_heads=2,d_ff=d_ff,norm='BatchNorm',bias=bias,
                 dropout=dropout,n_layers=n_layers,res_attention=False,pre_norm=pre_norm)
        
    def forward(self, x:torch.Tensor,ch_mask):
        """bs,max_ch,max_N,patch_len = x.shape ##the 'z' should be in the following order 
        x=x.view(bs,max_ch*max_N,-1)"""
        u=self.data_embedding(x)   
        #print(f'conv_output:{u.shape}')   
        z= self.encoder(u,ch_mask=ch_mask)
        ##z=torch.reshape(z,(-1,max_ch,max_N,self.d_model))
        return z  ##(bs,n_vars,num_patch,d_model)

#bool_mask_list.append(bool_mask)
"""
ts_data=torch.randn(1,1,1024)
actual_ch=ts_data.shape[1]
max_ch=21
lat_q_dim=5
bool_mask=torch.zeros(max_ch*lat_q_dim,dtype=torch.bool)
token_idx=torch.arange(actual_ch*lat_q_dim)
bool_mask[token_idx]=True
print(f'bolean_shape:{bool_mask.shape}')
bool_mask=bool_mask.unsqueeze(-1)
bool_mask_batch = torch.stack([bool_mask],dim=0)
print(bool_mask_batch.shape)
conv_layers_1=[(64,7,3,1),(128,5,3,2),(256,3,2,2),(512,3,2,2),(1024,3,2,2)]
ts_encoder= PatchTSTEncoder(conv_layers_1,1024,max_ch=21,n_layers=1,d_model=256,n_heads=2,d_ff=256,bias=False,lat_dim=5,
                 dropout=0.1,activation='gelu',pre_norm=False)
ts_embedding = ts_encoder(ts_data,bool_mask_batch)

print(f'final_embedding:{ts_embedding.shape}')"""
##print(ts_embedding[:86,:5])
##summary(ts_encoder,input_data=[ts_data,bool_mask])

