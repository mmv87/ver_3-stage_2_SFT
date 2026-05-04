import torch 
import torch.nn as nn  
from torch import Tensor
import torch.nn.functional as F
import torch.nn as nn
from typing import List, Tuple,Callable,Optional
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from conv_module import ConvFeatureExtractionModel
##from torchsummary import summary
from torchinfo import summary

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

##transpose util
class Transpose(nn.Module):
    def __init__(self, *dims,contiguous=False):
        super(Transpose, self).__init__()
        self.dims,self.contiguous= dims,contiguous
        
    def forward(self, x):
        if self.contiguous: return x.transpose(*self.dims).contiguous()
        else: return x.transpose(*self.dims)

##positional encoding
class positional_embedding(nn.Module):
    def __init__(self,max_N,max_channel,patch_len,device=None):
        super(positional_embedding,self).__init__()
        self.max_N=max_N ##for number of patches
        self.max_channel=max_channel
        self.embedding=patch_len
        self.device=device
        
        ## positional encoding for the patch position and channel indices
        self.patch_pos=nn.Embedding(self.max_N+1,self.embedding,padding_idx=self.max_N)  ###dictionary to store the positional information of the observation
        self.ch_pos = nn.Embedding(self.max_channel+1,self.embedding,padding_idx=self.max_channel) ## dictionary to store the embeddings for channel information
    
        self.patch_indices=torch.arange(self.max_N).unsqueeze(0).to(self.device) ### [0,1,2...max_N-1]
        self.channel_indices=torch.arange(self.max_channel).unsqueeze(0).to(self.device) ##[0,1,2...max_channel-1]
        ##print(self.patch_indices.shape)
        ###print(self.channel_indices.shape)

    def forward(self,x:torch.tensor,ts_token_mask:bool,ch_mask:bool):
        ##shape of x post conv_extraction: (bs,N,c_in,d)
        bs,max_ch,max_N,d = x.shape
        ###get the actual indices from the masks
        ##print(f'mask _shapes :{ts_token_mask.shape},{ch_mask.shape}')
        actual_idx = self.patch_indices[ts_token_mask]
        ##3print(actual_idx)
        actual_patch=self.channel_indices[ch_mask]
        ##print(actual_patch)
        ##get the positional embeddings for the patch positions
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
    
class ScaledDotProductAttention(nn.Module):
    def __init__(self,d_model,n_heads,attn_dropout=0.1,res_attention=False,lsa=True):
        super(ScaledDotProductAttention, self).__init__()
        self.attn_dropout=nn.Dropout(attn_dropout)
        self.res_attention = res_attention  ## boolean to use attention
        head_dim = d_model // n_heads
        self.scale = nn.Parameter(torch.tensor(head_dim ** -0.5), requires_grad=lsa)
        self.lsa=lsa ## boolean to use learnable scale

    def forward(self, q:Tensor, k:Tensor, v:Tensor,ts_mask:bool=None):
        attn_scores =torch.matmul(q,k)*self.scale
        
        if ts_mask is not None:
            attn_scores.masked_fill_(~(ts_mask.unsqueeze(0).unsqueeze(1)),float('-inf'))  ### resized the mask to the same as the attention matrix
        else:
            pass
        
        attn_weights =F.softmax(attn_scores, dim=-1)
        attn_weights= self.attn_dropout(attn_weights)##regularization
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
        
    def forward(self,Q:Tensor,K:Tensor,V:Tensor,mask=None):
        
        bs=Q.size(0)
        if K is None: K=Q
        if V is None:V=Q
        
        ##linear transformation of input tensor 'x' into Q,K and V
        q_s = self.W_Q(Q).view(bs,-1,self.n_heads,self.d_k).transpose(1,2) ##  (bs,n_heads,seq_len,d_k)
        k_s = self.W_K(K).view(bs,-1,self.n_heads,self.d_k).permute(0,2,3,1) ##  (bs,n_heads,seq_len,d_k)
        v_s = self.W_V(V).view(bs,-1,self.n_heads,self.d_v).transpose(1,2) ##  (bs,n_heads,seq_len,d_v)
        
        if self.res_attention:
            output,attn_scores,attn_weights= self.attention(q_s,k_s,v_s,ts_mask=mask)        
        else:
            output,attn_weights=self.attention(q_s,k_s,v_s,ts_mask=mask)
        ## output:[bs X n_heads X q_len X d_v]
        
        ## reassemble invidual heads to get MHSA output
        output= output.transpose(1,2).contiguous().view(bs,-1,self.n_heads*self.d_v) ## (bs,seq_len,n_heads*d_v) 
        
        output= self.to_out(output)
        
        if self.res_attention: 
            return output,attn_scores,attn_weights
        else: 
            return output,attn_weights 

class TS_encoder_layer(nn.Module):
    def __init__(self, d_model, n_heads,d_ff=256, store_attn = False , 
                 norm= 'BatchNorm',attn_dropout=0,dropout=0.,bias=True,activation='gelu',res_attention=True,pre_norm=False):
        
        super(TS_encoder_layer, self).__init__()
        ##self.d_model=d_model

        assert not d_model%n_heads, f'd_model ({d_model}) must be divisible by n_heads ({n_heads})'
        self.d_k = d_model // n_heads
        self.d_v = d_model // n_heads
        
        self.self_attn = MultiheadAttention(d_model,n_heads,d_k=self.d_k,d_v=self.d_v,res_attention=res_attention,
                 attn_dropout=attn_dropout,proj_dropout=0.1,qkv_bias=True,lsa=False)
        
        self.res_attention = res_attention
        self.dropout_attn= nn.Dropout(dropout)
        ##use Layer norm for attention sublayer pre-norm
        if 'batch' in norm.lower():
            self.norm_attn = nn.Sequential(Transpose(1,2),
                                            nn.BatchNorm1d(d_model),Transpose(1,2))
        else: 
            self.norm_attn = nn.LayerNorm(d_model)

        self.ff = nn.Sequential(nn.Linear(d_model,d_ff,bias=bias),
                                 nn.GELU(),
                                 nn.Dropout(dropout),
                                 nn.Linear(d_ff,d_model,bias=bias))
        
        ##add & norm 
        self.dropout_ffn=nn.Dropout(dropout)
        
        ##use Layer norm for FFN sublayer pre-norm
        if 'batch' in norm.lower():
            self.norm_ffn = nn.Sequential(Transpose(1,2),
                                           nn.BatchNorm1d(d_model),Transpose(1,2))
        else:
            self.norm_ffn = nn.LayerNorm(d_model)
            
        self.pre_norm = pre_norm ## optional to normalize prior to encoder block
        self.store_attn = store_attn
        
    def forward(self, src:Tensor, prev:Optional[Tensor]=None, key_padding_mask:Optional[Tensor]=None, attn_mask:Optional[Tensor]=None) -> Tensor:
        # pre-norm 
        """sns.kdeplot(src[0,3,:].detach().cpu().numpy())
        plt.show()"""
    
        if self.pre_norm:
            src = self.norm_attn(src)
        
        ##plot to check
        """sns.kdeplot(src[0,3,:].detach().cpu().numpy())
        plt.show()"""
        
        ##print(f'pre-norm_input:{src.shape}')
        ## 1. Multi-Head attention : sublayer-1
        if self.res_attention:
            src2, attn, scores = self.self_attn(src, src, src, prev, mask=attn_mask)
        else:
            src2, attn = self.self_attn(src, src, src, mask=attn_mask)
            
        if self.store_attn:
            self.attn = attn
        ## residual connection + dropout + norm
        src = src + self.dropout_attn(src2) # Add: residual connection with residual dropout (b*C,N,d_model)
        ###print(f'pre_shaping:{src[0,:,:5]}')
        ##multiply the ts_mask here to zero out the padded_tokens rows in the attention output 
        src = (src*attn_mask.to(src.dtype).unsqueeze(-1))   ##to match the dimension of attention output
        
        if not self.pre_norm:
            src = self.norm_attn(src)
            ## layer norm after attention + residual
        if self.pre_norm:
            src = self.norm_ffn(src)
            
        ##2. Position-wise Feed-Forward :sublayer-2
        src2 = self.ff(src)
        ##print(f'post_MLP_shape:{src2.shape}')

        ## Add & Norm
        src = src + self.dropout_ffn(src2) # Add: residual connection with residual dropout
        src = (src*attn_mask.to(src.dtype).unsqueeze(-1)) ## to zeros out the padded tokens after FFN
       
        ##print(f'post_MLP_masked_shape:{src[0,:,:5]}')
        
        if not self.pre_norm:
            src = self.norm_ffn(src)

        if self.res_attention:
            return src, scores
        else:
            return src

class TST_encoder(nn.Module):
    def __init__(self,d_model=None, n_heads=2,d_ff=256,norm='BatchNorm', attn_dropout=0., dropout=0., activation='gelu', n_layers=4, res_attention=False, pre_norm=False,store_attn=False):

        super(TST_encoder, self).__init__()
        
        self.layers = nn.ModuleList([
            TS_encoder_layer(d_model,n_heads,d_ff=d_ff,store_attn = store_attn,norm= norm,attn_dropout=attn_dropout,
                             dropout=dropout,bias=True,activation=activation,res_attention=res_attention,pre_norm=pre_norm)
            for _ in range(n_layers)
        ])

        self.res_attention = res_attention

    def forward(self, src: torch.Tensor, prev=None, mask=None):
        output=src ### initial input tensor
        scores=None
        
        if self.res_attention:
            for mod in self.layers:output,scores = mod(output, prev=scores, attn_mask=mask)
            return output
        
        else:
            for mod in self.layers: output = mod(output,attn_mask=mask)
            return output

class PatchTSTEncoder(nn.Module):
    def __init__(self,c_in=20,num_patch=10,patch_len=516,n_layers=1,d_model=128,n_heads=16,shared_embedding=True,d_ff=256,
                 norm='BatchNorm',attn_dropout=0.,dropout=0.1,activation='gelu',store_attn=False,res_attention=False,pre_norm=False,pe='zeros',learn_pe=True,verbose=False,**kwargs):
        
        super().__init__()
        self.n_vars = c_in
        self.num_patch = num_patch
        self.patch_len = patch_len   ##feature dimension of each patch
        self.d_model=d_model  ## llm embedding dimension
        self.shared_embedding = shared_embedding ## bool to have channel independence/mixing
        self.activation=activation
        self.n_heads=n_heads
        ##self.conv_layers=conv_layers
        ##self.conv_features=conv_features
        
        ##Convolutional_head for temporal encoding
        ##self.temporal_encoder=ConvFeatureExtractionModel(self.conv_layers,dropout=0.1)
        ##positional embedding
        self.W_pos=positional_embedding(self.num_patch,self.n_vars,self.patch_len,device=device)   ##adding timestep and channel information to the observation
        
        #input_embedding
        if not shared_embedding: ###channel independent embedding
            self.W_P=nn.ModuleList()
            for _ in range(self.n_vars):self.W_P.append(nn.Linear(patch_len,self.d_model))
        else:
            self.W_P=nn.Linear(patch_len,self.d_model)
            
        self.dropout=nn.Dropout(dropout)
        ##Encoder
        self.encoder = TST_encoder(d_model=self.d_model, n_heads=self.n_heads, d_ff=d_ff, norm=norm, attn_dropout=attn_dropout, dropout=dropout,
                                   pre_norm=pre_norm,activation=self.activation,res_attention=res_attention, n_layers=n_layers, store_attn=store_attn)
        
    def forward(self, x:torch.Tensor,ts_mask,ch_mask,ts_attention:bool=None):
        bs,max_ch,max_N,patch_len = x.shape ##the 'z' should be in the following order 
        ##z=self.temporal_encoder(x)
        ##_,_,_,feature_len=z.shape   ##(bs,n_vars,num_patch,conv_features)
        z=self.W_pos(x,ts_mask,ch_mask)  
        ### linear projection to d_model with optional shared embedding
        if not self.shared_embedding:
            x_out =[]
            for i in range(max_ch):
                z_proj=self.W_P[i](z[:,i,:,:]) ## (bs,num_patch,feature_len) -> (bs,num_patch,d_model)
                x_out.append(z_proj)
                
            z=torch.stack(x_out,dim=1) ## (bs,num_patch,d_model,n_vars)
            
        else: ### set this default option
            z=z.view(-1,max_N,patch_len) ### (bs*n_vars,num_patch,patch_len)
            z=self.W_P(z)   ##(bs*n_vars,num_patch,patch_len)
            print(f'after_linear_proj:{z.shape}')
            ##print(z[:,:,:5])
        
        u=z.view(-1,max_ch*max_N,self.d_model)    ## batch-first axis (bs,channels*num_patch,d_model)
        ##u=self.dropout(u+self.W_pos) ## dropout with positional encoding
        ## if only one patch, no need to pass through the encoder but with ts_padding tokens this becomes redundant
    
        z= self.encoder(u,mask=ts_attention) ##(bs*n_vars,num_patch,d_model) ## mask dimension[bs,total_tokens]
        z=torch.reshape(z,(-1,max_ch,max_N,self.d_model)) ## reshaped to (bs,n_vars,num_patch,d_model)

        return z  ##(bs,n_vars,num_patch,d_model)
"""     
##unit testing
test_x=torch.randn(1,5,10,512)
##print(test_x.shape)
max_ch=test_x.shape[1]
max_N=test_x.shape[2]
patch_len=test_x.shape[3]
##conv_layers = [(128,7,1),(128,7,1),(256,5,2),(256,5,2),(512,3,2)]
actual_N=3
actual_ch=1
ts_token_mask=((torch.arange(max_N).unsqueeze(0))<actual_N).bool().to(torch.device(device))
ch_mask=((torch.arange(max_ch).unsqueeze(0))<actual_ch).bool().to(torch.device(device))
print(f'ts_token_mask:{ts_token_mask}')
print(f'ch_mask:{ch_mask}')
"""

##total_mask=ts_token_mask.unsqueeze(1).expand(1,actual_ch,actual_N).contiguous().view(1,-1)  ## (bs,total_tokens)

##to get the global attention mask
##patch_copied =(ts_token_mask.long().expand(max_ch,ts_token_mask.shape[1]))
##patch_channel_adjusted = patch_copied[ch_mask]
##atten_mask_global =patch_copied.masked_fill(~(ch_mask.T.bool()),0)
##assert atten_mask_global.flatten().shape[0] == (max_ch*max_N)
##attention_mask =atten_mask_global.flatten().unsqueeze(0).bool()
###print(f'final_attn_mask :{attention_mask.shape}')
##print(attention_mask)

"""trans_ts_encoder=PatchTSTEncoder(c_in=5,num_patch=max_N,patch_len=patch_len,n_layers=2,d_model=768,n_heads=4,
                             shared_embedding=True,d_ff=256,norm='Layer',attn_dropout=0.,dropout=0.1,activation='gelu',store_attn=False,res_attention=False,pre_norm=True,pe='zeros',learn_pe=True,verbose=False)

ts_embedding =trans_ts_encoder(test_x,ts_token_mask,ch_mask,ts_attention=ts_token_mask)
print(ts_embedding.shape)"""
###print(ts_embedding[:,0,:,:5])
"""
conv_layers = [(256,5,2),(512,3,2)]
conv_ts_encoder=ConvFeatureExtractionModel(conv_layers,dropout=0.1)"""

"""conv_embeddings = conv_encoder(test_x)
print(conv_embeddings.shape)
concat_layer=torch.cat([conv_embeddings,ts_embedding],axis=-1)
print(concat_layer.shape)"""

##purpose to fuse and project the convolutional and ts_transformer into llm backbone
## wrapper to get the transformer and conv_module
"""class llm_projection(nn.Module):
    def __init__(self,conv_module,conv_features,trans_module,trans_embedding,d_fusion,d_llm):
        super().__init__()
        self.conv_module=conv_module
        self.trans_module=trans_module
        self.d_fusion=d_fusion
        self.d_llm=d_llm
        self.conv_features=conv_features
        self.trans_embedding=trans_embedding
        self.conv_proj=nn.Linear(self.conv_features,self.d_fusion)
        self.trans_proj=nn.Linear(self.trans_embedding,self.d_fusion)
        
        self.gate=nn.Linear(2*self.d_fusion,self.d_fusion)
        
        self.llm_projection=nn.Linear(self.d_fusion,self.d_llm)
        
    def forward(self,x,ts_mask,ch_mask,attention_mask):
        conv_embed= self.conv_module(x)
        trans_embed=self.trans_module(x,ts_mask,ch_mask,ts_attention=attention_mask)
        
        z_conv=self.conv_proj(conv_embed)
        z_trans=self.trans_proj(trans_embed)
        
        g=torch.sigmoid(self.gate(torch.cat([z_conv,z_trans],dim=-1)))
        z_gated=g*z_conv+(1.0-g)*z_trans
        z_llm = self.llm_projection(z_gated)
        
        return z_llm"""

"""
ts_encoder = llm_projection(conv_ts_encoder,512,trans_ts_encoder,768,1536,3072)

summary(ts_encoder,input_data=(test_x,ts_token_mask,ch_mask,attention_mask))

ts_embeddings = ts_encoder(test_x,ts_token_mask,ch_mask,attention_mask)"""

"""
total_params = sum(p.numel() for p in ts_encoder.parameters())
print(f"Total number of parameters: {(total_params/1e6):.2f} M")"""

"""print(ts_embeddings.shape)
print(ts_embeddings[:,0,:,:5])"""

        
    
        