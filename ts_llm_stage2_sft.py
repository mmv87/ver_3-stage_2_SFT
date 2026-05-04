import torch
import torch.nn as nn
import torch.nn.functional as F
##from TS_encoder import PatchTSTEncoder
from  transformers import AutoModelForCausalLM,AutoTokenizer
from transformers import get_cosine_schedule_with_warmup
from ts_dataloader_ import ts_textual,collate_func
import os
import sys
import numpy as np
from torch.utils.data import Dataset,DataLoader
from peft import get_peft_model 
from peft import LoraConfig, get_peft_model
from tqdm import tqdm
###modules for hybrid ts_encoder building
##from modules.conv_module import ConvFeatureExtraction
from modules.ts_encoder_rel_bias import PatchTSTEncoder
from modules.ts_encoder import llm_projection

device ='cuda' if torch.cuda.is_available() else 'cpu'

##location of llm_base model
model_name="/home/mmk/projects/def-zonata/mmk/hf_cache/hub/models--microsoft--Phi-4-mini-reasoning/snapshots/7a8c4e2e81eae20a606d811f475d7dc316dd916a"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
model = AutoModelForCausalLM.from_pretrained(model_name,local_files_only=True)
##expanded tokenizer path
tokenizer_path =os.path.join(os.environ["SLURM_TMPDIR"],'llm_tokenizer')
tokenizer_modified =AutoTokenizer.from_pretrained(tokenizer_path)
model_dtype=next(model.parameters()).dtype
## to expand the tokenizer to add the special tokens <ts> <ts/>
"""special_token_dict={'pad_token':"<|pad|>","additional_special_tokens":['<ts>','<ts/>']}
tokenizer.add_special_tokens(special_token_dict)"""
##model.resize_token_embeddings(len(tokenizer))
##dataset fetching
import json
_json_file = os.path.join(os.environ["SLURM_TMPDIR"],"sft_train.jsonl")
###datapipeline
dataset=ts_textual(21,5,tokenizer_modified,_json_file,15000,device=device)
dataloader=DataLoader(dataset,batch_size=1,shuffle=True,collate_fn=lambda b:collate_func(b,tokenizer=tokenizer_modified))
"""
dataset= ts_multimodal_text(128,128,_json_file,tokenizer,device=device,model_dtype=None)
dataloader=DataLoader(dataset,batch_size=1,shuffle=True,collate_fn=lambda b:collate_func(b,tokenizer=tokenizer,device=device))"""

##Lora_config defintion based on best practices
peft_config = LoraConfig(
            r=32, lora_alpha=32,
            target_modules=["o_proj",'qkv_proj','gate_up_proj','down_proj'],
            modules_to_save=["embed_tokens"],lora_dropout=0.1, # important for Stage-2  as to keep th ties
            task_type="CAUSAL_LM",ensure_weight_tying=True)

class LLM_wrapper(nn.Module):
    def __init__(self,tokenizer,conv_layers,llm_model,lat_dim,device=device,ts_checkpoint=None,embed_path=None,peft_config=None):
        super().__init__()
        self.lat_dim=lat_dim
        self.tokenizer=tokenizer
        self.llm_model=llm_model
        self.embed_size=llm_model.config.hidden_size
        #self.P=patch_len
        self.device=device
        self.conv_layers=conv_layers
        self.peft_config=peft_config
        ##resize the input_embedding_layer
        self.llm_model.resize_token_embeddings(len(self.tokenizer))
        
        if embed_path:
            self.llm_model.get_input_embeddings().load_state_dict(torch.load(embed_path))
        ###creating peft model for stage-2 training
        if self.peft_config:
            self.peft_model=get_peft_model(self.llm_model,self.peft_config)
            self.peft_model.train()
        print(f"Embeddings trainable:{self.peft_model.base_model.model.model.embed_tokens.weight.requires_grad}")
   
        """  if "embed_tokens" in self.peft_config.modules_to_save:
                embed_layer = self.peft_model.get_input_embeddings() 
                
            if hasattr(embed_layer, "modules_to_save"):
                embed_layer.modules_to_save.default.weight.requires_grad_(True)
            else:
                embed_layer.requires_grad_(True)

            # CRITICAL: Re-tie the weights to ensure Stage-2 updates the Manifold    
            input_wrapper=self.peft_model.get_output_embeddings()
            output_wrapper=self.peft_model.get_input_embeddings()
            
            if hasattr(input_wrapper, "modules_to_save") and hasattr(output_wrapper, "modules_to_save"):
                # Target the actual trainable weight tensor in the input
                trainable_weight = input_wrapper.modules_to_save.default.weight
                # Target the 'default' module inside the output wrapper
                target_module = output_wrapper.modules_to_save.default
                # Surgical re-tie: Delete the pointer and re-register
                # We do this on the .default module, which is a standard nn.Linear/nn.Embedding
                if hasattr(target_module, "weight"):
                    delattr(target_module, "weight")
                    
                target_module.register_parameter("weight", trainable_weight)
              
            # If this prints 'True', your Stage-2 alignment will physically move the manifold
            assert id(self.peft_model.get_input_embeddings().weight) == id(self.peft_model.get_output_embeddings().weight)
            print("Weight tie successfully established via register_parameter.")"""
            ##print(f"Tied: {id(self.peft_model.get_input_embeddings().weight) == id(self.peft_model.base_model.model.lm_head.weight)}")
            
        #self.ts_conv_module=ConvFeatureExtraction(self.conv_layers,dropout=0.1)
        self.ts_transformer = PatchTSTEncoder(self.conv_layers,1024,max_ch=21,n_layers=1,d_model=256,n_heads=2,d_ff=256,bias=False,lat_dim=self.lat_dim,
                 dropout=0.1,activation='gelu',pre_norm=True,device=self.device)
        
        self.ts_encoder = llm_projection(self.ts_transformer,1024,2048,3072,device=self.device)
        
        ##ts_encoder state_dict loading
        ts_enc_state_dict = torch.load(ts_checkpoint, map_location=self.device)
        self.ts_encoder.load_state_dict(ts_enc_state_dict,strict=False)
        self.ts_encoder.to(self.device)
        
        for p in self.ts_encoder.parameters():
            p.requires_grad = True
        
    def assemble_input_embeds(self,input_ids,ts_embeddings,ts_token_idx,text_token_idx,ts_pairs:torch.tensor):
        ###logic to assemble textual and ts_tokens 
        assemb_embed_tensor=[]
        channels=ts_pairs.shape[1]
    
        ##slicing the ts_embedding
        slicing_dim=channels*self.lat_dim
        ts_embeddings_slice=torch.narrow(ts_embeddings,1,0,slicing_dim)
        ts_embeddings_slice.to(self.device) ##to make sure it is in same device
        
        bs=ts_embeddings_slice.shape[0]
        T=ts_embeddings_slice.shape[1]
        ts_emb_dim=ts_embeddings_slice.shape[2]
        assert T==slicing_dim
     
        ##ts_embeddings=ts_embeddings.view(bs*c_in,num_ts_tokens,-1)        
        embed_module=self.peft_model.get_input_embeddings()  ##[bs,seq_len,d_emb]
        ##explicitly setting the embed from the default
        if hasattr(embed_module, "modules_to_save"):
            input_embeds = embed_module.modules_to_save.default(input_ids)
        else:
            # Fallback if PEFT isn't initialized or for base mode
            input_embeds = embed_module(input_ids)
        
        print(f'input_embeds_shape:{input_embeds.shape}')
        ###print(f'input_embeds_shape:{input_embeds.shape}')
        ###input_embeds.requires_grad_(requires_grad=True)
        text_emb_dim= input_embeds.shape[2]
        assert (ts_emb_dim==text_emb_dim)
        T_new=ts_token_idx.shape[1]+text_token_idx.shape[1]
        ts_container =torch.zeros((T_new,text_emb_dim),device=self.device) ### total_idx,total_idx
        text_container=torch.zeros((T_new,text_emb_dim),device=self.device)
        flat_ts_embeddings=ts_embeddings_slice.view(-1,T,ts_emb_dim)
        flat_ts_embeddings=flat_ts_embeddings.squeeze(0)
        ##print(f'ts_embedding_flat:{flat_ts_embeddings.shape}')
        
        flat_text_embeddings=input_embeds.squeeze(0)
        ##get the indices after the <ts>....<ts/> placeholder is offseted
        ts_indices=ts_token_idx.squeeze(0).view(-1,1)
        ts_indices=ts_indices.expand(-1,text_emb_dim)
        text_indices=text_token_idx.squeeze(0).view(-1,1)
        text_indices=text_indices.expand(-1,text_emb_dim)
       
        ts_embeds_assemb= ts_container.scatter(dim=0,index=ts_indices,src=flat_ts_embeddings)
        final_tensor=ts_embeds_assemb.scatter(dim=0,index=text_indices,src=flat_text_embeddings)
        print(f'final_tensor:{final_tensor.shape}')
        #final_tensor=ts_embeds_assemb+text_embeds_assemb
        assemb_embed_tensor.append(final_tensor)
        
        return torch.stack(assemb_embed_tensor)

    def forward(self,input_ids=None,ts_input=None,ts_pairs=None,ts_idx=None,text_idx=None,attention_mask=None,ch_mask=None,labels=None,):
        ##convert the ts_patches into ts_embeddings
        ts_tensor = ts_input.to(self.device)  ## (bs,c_in,N,P)
        ts_embedding = self.ts_encoder(ts_tensor.to(self.device),ch_mask) ## (bs,n_vars,num_patch,d_model)
        ##slicing
        ##ts_embedding_sliced =ts_embedding[ts_masks] ##flattened ts_embeddings
        input_embeddings= self.assemble_input_embeds(input_ids,ts_embedding,ts_idx,text_idx,ts_pairs)
        ##print(f'input_embeddigs:{input_embeddings.shape}')
        attention_mask = attention_mask.to(self.device)
        labels = labels.to(self.device)
        ##print(f'labels:{labels.shape}')
        output= self.peft_model(inputs_embeds=input_embeddings,attention_mask=attention_mask,labels=labels)
        
        return output,input_embeddings
    
##load the pre-trained weights
ts_encoder_weights=os.path.join(os.environ["SLURM_TMPDIR"],'ts_enc_stage1_ver3.pth')
embed_path=os.path.join(os.environ["SLURM_TMPDIR"],'aligned_embeddings.pt')

from tqdm import tqdm
conv_layers_1=[(64,7,3,1),(128,5,3,2),(256,3,2,2),(512,3,2,2),(1024,3,2,2)]
model_wrapper=LLM_wrapper(tokenizer_modified,conv_layers_1,model,5,device=device,ts_checkpoint=ts_encoder_weights,embed_path=None,peft_config=peft_config)
model_wrapper.train()
model_wrapper.to(device)

####check the gradient

def check_input_emb(peft_model):
    emb_layer = peft_model.get_input_embeddings()
    # 2. Check if it's a PEFT wrapper (ModulesToSaveWrapper)
    # If so, we want the 'default' (trainable) module, not 'original_module'
    if hasattr(emb_layer, "modules_to_save"):
        active_emb = emb_layer.modules_to_save.default
    else:
        active_emb = emb_layer
        
    # 3. Probe the gradient
    if active_emb.weight.grad is not None:
        with torch.no_grad():
            # Global norm for the whole layer
            total_norm = active_emb.weight.grad.norm(2).item()
            # Row-wise norm to see active tokens (as discussed)
            per_token_norms = torch.linalg.vector_norm(active_emb.weight.grad, ord=2, dim=1)
            active_tokens = (per_token_norms > 0).sum().item()
            print(f"--- Embedding Check ---")
            print(f"Total Grad Norm: {total_norm:.6f}")
            print(f"Active Tokens in Batch: {active_tokens}")
    else:
        print("CRITICAL: Active embedding layer has NO gradient. Check if loss.backward() was called.")
    
def check_ts_gradients(ts_encoder):
    print("\n--- Gradient Flow Check: TS Encoder ---")
    any_grad = False
    for name, param in ts_encoder.named_parameters():
        if not param.requires_grad:
            print(f"{name}: Frozen (requires_grad=False)")
            continue
        if param.grad is None:
            print(f"{name}: Grad is None (Graph Broken!)")
        else:
            grad_norm = param.grad.norm().item()
            print(f"{name}: Grad Norm = {grad_norm:.5f}")
            if grad_norm > 1e-6:
                any_grad = True
                
    if not any_grad:
        print("WARNING: No trainable parameters in TS Encoder received gradients.")
    else:
        print("Success: Gradients are flowing to TS Encoder.")

###different learing rates for enc and llm_model
for name, param in model_wrapper.ts_encoder.named_parameters():
    param.requires_grad = True
    
encoder_params = list(model_wrapper.ts_encoder.parameters())
llm_trainable_params = [p for n,p in model_wrapper.peft_model.named_parameters() if p.requires_grad]

optimizer = torch.optim.AdamW([
    {'params': encoder_params, 'lr': 1e-4,'weight_decay':0.05},      # Physics: Fast learning
    {'params': llm_trainable_params, 'lr': 1e-5,'weight_decay':0.01}]) ##slow learning lm 

##** freeze the LLM for stage-1 training
epoch_losses=[]
num_epochs=2
###adaptive learning rate
steps_per_epoch = len(dataloader) # This is Total Samples / Batch Size
total_steps = num_epochs * steps_per_epoch
num_warmup_steps = int(0.1 * total_steps)

###schedulers
scheduler = get_cosine_schedule_with_warmup(
    optimizer,  
    num_warmup_steps=num_warmup_steps,
    num_training_steps=total_steps
)

for epoch in range(num_epochs):  ##1 epochs
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    num_batches = 0
    running_loss=0
    epoch_loss=0
    ctr=0
    for batch in pbar:
        input_ids=batch['input_ids'].to(device) ## input and output
        labels_batch=batch['labels'].to(device)
        attention_mask=batch['attention_mask'].to(device)
        ts_input=batch['time_series'].to(device) ### batch of patchified padded ts_inputs (bs,c_in,N,p)
        ts_pairs=batch['ts_pairs'].to(device)
        ts_indices=batch["ts_indices"].to(device)
        textual_indices=batch['textual_indices'].to(device)
        ch_mask =batch['ch_mask'].to(device)
        ###ts_mask = batch['ts_mask'].to(device)
        ##model_wrapper=LLM_wrapper(tokenizer,ts_input,model,device=device)
        optimizer.zero_grad()
        outputs,_= model_wrapper(input_ids=input_ids,ts_input=ts_input,ts_pairs=ts_pairs,ts_idx=ts_indices,text_idx=textual_indices,
                                 attention_mask=attention_mask,ch_mask=ch_mask,labels=labels_batch,)
        loss=outputs.loss
        loss.backward()  
        check_ts_gradients(model_wrapper.ts_encoder)##gradient calculation
        #check_input_emb(model_wrapper.peft_model)
        running_loss+=loss.item()
        num_batches+=1
        optimizer.step()
        scheduler.step()
        ###gradient checking
        pbar.set_postfix(loss=loss.item())
        epoch_loss=running_loss/num_batches
        epoch_losses.append(epoch_loss)
        ###ctr+=1
##x=len(epoch_losses)
###save the ts_encoder and the trained llm adapters
saved_file=os.path.join(os.environ["SLURM_TMPDIR"],'ts_encoder_ver2_final.pth')
torch.save(model_wrapper.ts_encoder.state_dict(),saved_file)

##model_wrapper.peft_model.config.save_embedding_layers = True
##save the llm_model
save_path=os.path.join(os.environ["SLURM_TMPDIR"],'phi4-ts-adapter_ver3')
model_wrapper.peft_model.save_pretrained(save_path,safe_serialization=True,save_embedding_layers=True)
"""saved_file=os.path.join(os.environ["SLURM_TMPDIR"],'ts_enc_stage1_ver2.pth')
torch.save(model_wrapper.ts_encoder.state_dict(),saved_file)
###embedding layer 
embeds = model_wrapper.llm_model.get_input_embeddings().state_dict()
torch.save(embeds, os.path.join(os.environ["SLURM_TMPDIR"], "aligned_embeddings_ver2.pt"))
##tokenizer saved
tokenizer.save_pretrained(os.path.join(os.environ["SLURM_TMPDIR"],'llm_tokenizer'))"""
### save the plot
out_path = os.path.join(os.environ["SLURM_TMPDIR"], "training_loss_ver2.png")
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

plt.figure(figsize=(10,15))
plt.plot(epoch_losses)
plt.title("Training Loss Trend Over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Average Loss")
plt.grid(True)
plt.savefig(out_path)