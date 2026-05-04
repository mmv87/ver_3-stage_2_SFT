
###Main ts_encoder that fuses the conv_module and transformer 
###Concatenate the features from conv_module and ts_transformer module and project to the llm_backbone 
## modify to z_conv as information filter 
"""from conv_module import ConvFeatureExtraction
from ts_encoder_rel_bias import PatchTSTEncoder
from ts_encoder_rel_bias import PatchTSTEncoder"""
import torch.nn as nn  
import torch
#from modules.ts_encoder_perceiver_resampler import PatchTSTEncoder
device ='cuda' if torch.cuda.is_available() else 'cpu'

class llm_projection(nn.Module):
    def __init__(self,trans_module,trans_embedding,d_fusion,d_llm,device=None):
        super().__init__()
        self.device =device
        self.ts_encoder=trans_module
        
        self.trans_embedding=trans_embedding #1024
        self.d_fusion=d_fusion ###intermediate dimesnion to 512-->[1024]-->3072
        self.d_llm=d_llm

        ###multimodal_bridge to project from z_trans to llm_embed
        self.mm_bridge=nn.Sequential(
            nn.Linear(self.trans_embedding,self.d_fusion,bias=False),
            nn.GELU(),
            nn.Linear(self.d_fusion,self.d_llm,bias=False)
        )
        self.norm_projection=nn.LayerNorm(self.d_llm)
        
    def forward(self,x,ch_mask=None):
        ts_features= self.ts_encoder(x,ch_mask)
        ts_features.to(self.device)
        
        z_proj=self.mm_bridge(ts_features)
        ##z_gated=z_trans*z_mask
        ##layer norm
        z_llm = self.norm_projection(z_proj)

        return z_llm
"""
ts_test=torch.randn(1,1,1014)
actual_ch=ts_test.shape[1]
max_ch=21
lat_q_dim=5
bool_mask=torch.zeros(max_ch*lat_q_dim,dtype=torch.bool)
token_idx=torch.arange(actual_ch*lat_q_dim)
bool_mask[token_idx]=True
print(f'bolean_shape:{bool_mask.shape}')
bool_mask=bool_mask.unsqueeze(-1)
bool_mask_batch = torch.stack([bool_mask],dim=0)
print(bool_mask_batch)

conv_layers_1=[(64,7,3,1),(128,5,3,2),(256,3,2,2),(512,3,2,2),(1024,3,2,2)]
ts_encoder= PatchTSTEncoder(conv_layers_1,1024,max_ch=21,n_layers=1,d_model=256,n_heads=2,d_ff=256,bias=False,lat_dim=5,
                 dropout=0.1,activation='gelu',pre_norm=False)

ts_encoder = llm_projection(ts_encoder,1024,2048,3072,device=device)
ts_embeddings=ts_encoder(ts_test,ch_mask=bool_mask_batch)

print(ts_embeddings.shape)

ts_embedding_new=torch.narrow(ts_embeddings,1,0,actual_ch*lat_q_dim)
print(ts_embedding_new)"""
##print(ts_embeddings[:,:7,:5])

"""
for param_tensor in ts_encoder.state_dict():
    if 'trans' in param_tensor:
        print(f'{param_tensor} :{ts_encoder.state_dict()[param_tensor].size()}')
    else:
        continue """
        



