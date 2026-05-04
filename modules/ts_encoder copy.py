
###Main ts_encoder that fuses the conv_module and transformer 
###Concatenate the features from conv_module and ts_transformer module and project to the llm_backbone 
## modify to z_conv as information filter 
"""from conv_module import ConvFeatureExtraction
from ts_encoder_rel_bias import PatchTSTEncoder
from ts_encoder_rel_bias import PatchTSTEncoder"""
import torch.nn as nn  
import torch
class llm_projection(nn.Module):
    def __init__(self,conv_module,conv_features,trans_module,trans_embedding,d_fusion,d_llm):
        super().__init__()
        self.conv_module=conv_module
        self.trans_module=trans_module
        self.d_fusion=d_fusion ###intermediate dimesnion to 512-->[1024]-->3072
        self.d_llm=d_llm
        self.conv_features=conv_features
        self.trans_embedding=trans_embedding

        #filter based on z_conv
        self.filter_gen = nn.Sequential(
            nn.Linear(self.conv_features, self.trans_embedding),
            nn.LayerNorm(self.trans_embedding),
            nn.Sigmoid()
        )
        ###multimodal_bridge to project from z_trans to llm_embed
        self.mm_bridge=nn.Sequential(
            nn.Linear(self.trans_embedding,self.d_fusion),
            nn.GELU(),
            nn.Linear(self.d_fusion,self.d_llm)
        )
        ###self.gate=nn.Linear(2*self.d_fusion,self.d_fusion)
        ###self.llm_projection=nn.Linear(self.d_fusion,self.d_llm)
        self.norm_projection=nn.LayerNorm(self.d_llm)
        
    def forward(self,x):
        conv_embed= self.conv_module(x)
        trans_embed=self.trans_module(x)
        ##z_conv=self.conv_proj(conv_embed)
        ##z_conv to create a mask.
        z_mask=self.filter_gen(conv_embed)
        ##print(z_mask.shape)
        z_gated=z_mask*trans_embed
        z_proj=self.mm_bridge(z_gated)
        ##z_gated=z_trans*z_mask
        ##layer norm
        z_llm = self.norm_projection(z_proj)
        return z_llm


"""
ts_text=torch.randn(1,3,2,128)
conv_layers = [(128,5,1),(64,3,1)]
patch_len=128
ts_transformer=PatchTSTEncoder(patch_len=patch_len,n_layers=2,d_model=512,n_heads=8,
                             shared_embedding=True,d_ff=1024,norm='Layer',attn_dropout=0.,dropout=0.1,activation='gelu',store_attn=False,res_attention=False,pre_norm=True,pe='zeros',learn_pe=True,verbose=False)
ts_conv_module=ConvFeatureExtraction(conv_layers,dropout=0.1)

ts_encoder = llm_projection(ts_conv_module,64,ts_transformer,512,1024,3072)
"
for param_tensor in ts_encoder.state_dict():
    if 'trans' in param_tensor:
        print(f'{param_tensor} :{ts_encoder.state_dict()[param_tensor].size()}')
    else:
        continue 

ts_embeddings=ts_encoder(ts_text)
print(ts_embeddings.shape)"""


