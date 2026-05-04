###Datapipeline for SFT.jsonl suitable for multi-variate and univariate timeseries
## Updated with the attention_mask (accounting for ts_token padding)
## dataloader to set the pipeline
### for the subset of the dataset
import os
###os.environ['HF_HOME']='D:/hf_cache'
from torch.utils.data import Dataset,DataLoader
import torch
import json
from transformers import AutoModelForCausalLM,AutoTokenizer
import numpy as np
from torch.nn.utils.rnn import pad_sequence
device ='cuda' if torch.cuda.is_available() else 'cpu'

abs_modelpath="D:/hf_cache/hub/models--microsoft--Phi-4-mini-reasoning/snapshots/0e3b1e2d02ee478a3743abe3f629e9c0cb722e0a"
##print('path_read')
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
model_name='./hub/microsoft/phi-4-mini-reasoning'
device ='cpu'
print(device)

"""model=AutoModelForCausalLM.from_pretrained(abs_modelpath,local_files_only=True)
model.to(device)"""

tokenizer=AutoTokenizer.from_pretrained(abs_modelpath,local_file_only=True)
###add special_tokens to the tokenizer
special_token_dict={'pad_token':"<|pad|>","additional_special_tokens":['<ts>','<ts/>']}
tokenizer.add_special_tokens(special_token_dict)
align_256_file='D:/Doctoral_research/code_implementation/Time_series_reasoning/training_dataset/ChatTS-Training-Dataset/align_256/train.jsonl'
sft_file='D:/Doctoral_research/code_implementation/Time_series_reasoning/training_dataset/ChatTS-Training-Dataset/sft/sft_train.jsonl'
#ift_dataset ="D:/Doctoral_research/code_implementation/dataset/ChatTS-Training-Dataset/ift/train.jsonl"

## Dataset class to get the pipeline for a sample

class ts_textual(Dataset): 
    def __init__(self,max_ch,lat_dim,tokenizer,file,sample_size,device=device):
        super().__init__()
        self.max_ch=max_ch
        self.lat_dim=lat_dim
        self.tokenizer=tokenizer
        self.file=file
        self.device =device
        self.sample_size=sample_size
        self.byte_offset=[]
        
        with open(self.file,'rb') as f:
            while True:
                current_pos=f.tell()
                line = f.readline()
                if not line:
                    break
                if line.strip():
                    try:
                        self.byte_offset.append(current_pos)
                    except:
                        print('error in the line')
        
        self.sliced_offset=self.byte_offset[:self.sample_size]

    def __len__(self):
        return len(self.sliced_offset)
    
    def sp_encoding(self,timeseries):
        ##logic to get the normalize and get the 
        meta_prompts=[]
        timeseries_list=[]
        for ts_data in timeseries:
            mean = np.mean(ts_data)
            scaled_timeseries = ts_data - mean
            scale_factor = 1.0
            if np.any(np.abs(scaled_timeseries) >= 3.0):
                scale_factor = np.max(np.abs(scaled_timeseries)) / 3.0
                scaled_timeseries /= scale_factor
            # meta-prompt
            meta_prompt = f"[Value Offset: {mean:.4f}|Value Scaling: {scale_factor:.4f}]"
            meta_prompt_tokens=self.tokenizer(meta_prompt,return_tensors='pt')['input_ids']
            ##print(f'meta_shape{meta_prompt_tokens.shape}')
            meta_prompts.append(meta_prompt_tokens)
            list_ts=scaled_timeseries.tolist()
            ###print(f'list_ts:{len(list_ts)}')
            timeseries_list.append(list_ts)
            
        ###print(f'meta_prompts:{len(meta_prompts)}')
        return timeseries_list,meta_prompts
    
    def insert_meta_prompt(self,sequence:torch.Tensor,meta_prompts:list,ts_start):
        current_offset = 0
        result = sequence.clone() if torch.is_tensor(sequence) else list(sequence)
        result.unsqueeze_(0)
        ###print(f'result_seq:{result.shape}')
        ts_start_indices=ts_start.tolist()
        
        for i, original_pos in enumerate(ts_start_indices):
            ##print(f'orginal_ts_start:{original_pos}')
            actual_pos = original_pos + current_offset
            # Get the specific meta_prompt for this channel/pair
            meta = meta_prompts[i]
            meta_len = meta.shape[1]
            # Perform Splice
            if torch.is_tensor(result):
                result = torch.cat([result[:,:actual_pos],meta,result[:,actual_pos:]],dim=1)
            else:
                result = result[:,:actual_pos] + meta + result[:actual_pos:]
            # Update the offset for the NEXT iteration
            current_offset += meta_len
        
        #print(f'total_textual_len:{result.shape[1]}')
        return result,result.shape[1]
    
    def pad_and_patchify(self,ts_input:list):
        seq_len_list=[]
        #pad_pattern=torch.tensor([0.0,0.0],dtype=torch.float16)
        if len(ts_input)>1 : ##multivariable case
            ##check if the individual tensors are same shape
            for metric in ts_input:
                seq_len_list.append(torch.tensor(metric).shape[0]) ###get the list of tensors 
                ##print(torch.tensor(metric).shape)
            if max(seq_len_list)!=min(seq_len_list):
                ##print('staggered')
                print(max(seq_len_list),min(seq_len_list))
                ##ts_type='staggered' 
                ts_padded_list=[]
                ###remove the stagger
                for metric in ts_input:
                    ts_padded_list.append(torch.tensor(metric))
                ts_uniform=pad_sequence(ts_padded_list,batch_first=True,padding_value=0)
                ch_dim=ts_uniform.shape[0]
                seq_len=ts_uniform.shape[1]
                assert ts_uniform.shape[1]==max(seq_len_list)
            else:
                print('uniform')
                ts_tensor=torch.tensor(ts_input)
                ##print(ts_tensor.shape)
                ###ts_tensor.unsqueeze_(-1)
                ##print(f'ts_tensor_shape:{ts_tensor.shape}')
                seq_len=ts_tensor.shape[1]
                ch_dim=ts_tensor.shape[0]
                ts_uniform=ts_tensor.clone()
                ###print(f'seq_len:{seq_len}')
               
                ##return ts_patched
        else:                ##univariate case
            print('univariate')
            ts_tensor=torch.tensor(ts_input)
            ts_uniform=ts_tensor.clone()
            seq_len=ts_tensor.shape[1]
            ch_dim=ts_tensor.shape[0] ###ch=1 i.e univariate
            ##pad_width=(seq_len-p)%s
            
            ##return ts_patched
            
        return ts_uniform       
    
    def ts_pair_indices(self,tokenized,prefix):
        ts_start_token=self.tokenizer.convert_tokens_to_ids('<ts>')
        ts_end_token=self.tokenizer.convert_tokens_to_ids('<ts/>')
        ts_position=[]
        
        ##logic to ignore<ts><ts/> in the prefix prompt
        if ts_start_token in prefix:
            start_offset=True
        else:
            start_offset=False
        print(start_offset)
        for i,token_id in enumerate(tokenized.tolist()):
            if (token_id==ts_start_token):
                ts_position.append(('start',i))
            elif (token_id==ts_end_token):
                ts_position.append(('end',i))
        stack =[]
        ts_pairs=[]
        
        for j in range(len(ts_position)):
            pos,idx = ts_position[j]
            if pos=='start':
                stack.append(idx)
            elif stack and pos=='end':
                start=stack.pop(0)
                ts_pairs.append((start,idx))
                
        if start_offset:
            ts_pairs=ts_pairs[1:]
        else:pass

        return ts_pairs,tokenized.shape[0] ##list of tuples
     
    def _calculate_ts_indices(self,ts_pairs,c_in,max_N,total_textual_tokens):
        ##to calculate the ts_indices and textual indices for a sample
        tensor_ts_pairs=(torch.tensor(ts_pairs))
        channel_indices=torch.arange(c_in,dtype=torch.long)
        ##offset_vec = (channel_indices*3)
        tensor_ts_pairs[:,:]+=channel_indices.view(-1,1)*max_N
        tensor_ts_pairs[:,1]+=max_N
        new_ts=(tensor_ts_pairs[:,0])
        offset_entries=(torch.arange(1,max_N+1).view(-1,1))
        ts_indices=(new_ts+offset_entries).t().flatten() ### indices for ts_patch insertions
        ###total_indices=torch.arange(1,40)
        tot_multimodal_tokens=total_textual_tokens+(c_in*max_N)
        is_ts_new=torch.zeros(tot_multimodal_tokens, dtype=torch.bool)
        is_ts_new[ts_indices]=True
        new_text_indices = torch.nonzero(~is_ts_new).squeeze()
        
        return ts_indices,new_text_indices,tot_multimodal_tokens
        
    def __getitem__(self,idx):
        ##self.byte_offset[idx]
        with open(self.file,'rb') as file:
            file.seek(self.sliced_offset[idx])
            line =file.readline()
            sample =json.loads(line)

        input = sample['input']
        output = sample['output']
        timeseries=sample['timeseries'] ###list of lists
        
        prefix_prompt=f"<|system|>You are timeseries analyst,based on the context and the timeseries embeddings inside the <ts><ts/> tags generate the suitable answer<|end|>"
        prompt=f"<|user|>{input}<|end|>"
        output_prompt=f"<|assistant|>{output}<|end|>"
        
        prefix_ids=self.tokenizer(prefix_prompt,return_tensors='pt',add_special_tokens=False)['input_ids'][0]
        input_ids=self.tokenizer(prompt,return_tensors='pt',add_special_tokens=False)['input_ids'][0]
        output_ids=self.tokenizer(output_prompt,return_tensors='pt',add_special_tokens=False)['input_ids'][0]
        
        ###total_textual_ids
        combined_ids=torch.cat([prefix_ids,input_ids,output_ids],dim=0)
        ##normalize the ts_data
        norm_ts,meta_prompt = self.sp_encoding(timeseries)
        ts_pairs,text_tokens_pre_meta_prompt=self.ts_pair_indices(combined_ids,prefix_ids)
        ts_start=torch.tensor(ts_pairs)[:,0]   
        ##print(f'ts_start:{ts_start}')     
        new_text_tokens,total_text_tokens=self.insert_meta_prompt(combined_ids,meta_prompt,ts_start)
        ts_uniform =self.pad_and_patchify(norm_ts)
        ts_data=torch.tensor(ts_uniform)
        
        ch=ts_data.shape[0]
        assert len(ts_pairs)==ch
        ts_tokens,text_tokens,total_multlimodal_tokens=self._calculate_ts_indices(ts_pairs,ch,self.lat_dim,total_text_tokens)
        ##labels
        output_len=output_ids.shape[0]
        labels = torch.full((total_multlimodal_tokens,),-100,dtype=torch.long,device=self.device)
        labels[-output_len:] = output_ids.clone()
        ###assert labels.shape==combined_ids.shape
        ##attention_mask
        attention_mask=torch.ones(total_multlimodal_tokens,dtype=torch.long,device=self.device)
        ##attention_mask_batch.append(attention_mask)  
        bool_mask=torch.zeros(self.max_ch*self.lat_dim ,dtype=torch.bool)
        token_idx=torch.arange(ch*self.lat_dim)
        bool_mask[token_idx]=True
        bool_mask.to(self.device)
                  
        return{"input_ids":new_text_tokens,
            "output_ids":output_ids,
            "ts_input":ts_data,
            "labels":labels,
            "attention_mask":attention_mask,
             "ts_indices":ts_tokens,
             "text_indices":text_tokens,
             "ts_pairs":torch.tensor(ts_pairs),
             "ch_mask":bool_mask.unsqueeze(-1)}

###collate function
def collate_func(batch,tokenizer=None):
    input_ids = [x['input_ids'] for x in batch]
    labels_batch=[x['labels'] for x in batch]
    attention_mask_batch=[x['attention_mask'] for x in batch]
    padded_ts_data=[x['ts_input'] for x in batch] 
    ts_pairs=[x['ts_pairs'] for x in batch]
    ###assembler helper vars
    ts_indices =[x['ts_indices'] for x in batch] 
    text_indices=[x['text_indices'] for x in batch]
    ch_mask_batch=[x['ch_mask'] for x in batch]
    
    return{
        'input_ids':torch.cat(input_ids),
        "labels":torch.stack(labels_batch),
        'attention_mask':torch.stack(attention_mask_batch),
        "time_series":torch.stack(padded_ts_data),
        "ts_indices":torch.stack(ts_indices),
        "textual_indices":torch.stack(text_indices),
        "ts_pairs":torch.stack(ts_pairs),
        "ch_mask":torch.stack(ch_mask_batch,dim=0)}   ##list of tensor (bs,max_N,Patch_len)

###dataset=ts_textual(128,128,_json_path,tokenizer_modified,device=device,model_dtype=None)
##dataloader

dataset_for_test=ts_textual(21,5,tokenizer,sft_file,10000,device=device)
dataloader=DataLoader(dataset_for_test,batch_size=1,shuffle=True,collate_fn=lambda b:collate_func(b,tokenizer=tokenizer))
#input_embeds = model.get_input_embeddings()

for idx,batch in enumerate(dataloader):
    if idx<10000:
        print(batch['time_series'].shape)
        #print(tokenizer.decode(batch['input_ids'][0]))
        print(f"input_ids:{batch['input_ids'].shape}")
        print(f"ch_mask:{batch['ch_mask'].to(torch.int16).sum(1)}")
        print(f"ts_indices:{batch['ts_indices'].shape}")
        ###text_embedding = input_embeds(batch['input_ids'])
        ###print(f'textual_embedding{text_embedding.shape}')
        print(batch['labels'].shape)
        print(batch['attention_mask'].shape)
    else:
        break