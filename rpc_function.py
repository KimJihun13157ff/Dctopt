import os
import torch
import torch.distributed.rpc as rpc
from torch.distributed.rpc import RRef
import numpy as np
import random
import time 

import wandb
from torch.utils.data import DataLoader
import global_vars
 ## Todo
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def worker_set_up(args, pidx, client_list, datasets):
    global_vars.Client_list = client_list
    global_vars.Process_index = pidx 
    
    
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    tokenizer.model_max_length = args.max_length
    special_tokens = dict()
    if tokenizer.pad_token is None:
        special_tokens["pad_token"] = DefaultToken.PAD_TOKEN.value
    if tokenizer.eos_token is None:
        special_tokens["eos_token"] = DefaultToken.EOS_TOKEN.value
    if tokenizer.bos_token is None:
        special_tokens["bos_token"] = DefaultToken.BOS_TOKEN.value
    if tokenizer.unk_token is None:
        special_tokens["unk_token"] = DefaultToken.UNK_TOKEN.value
    tokenizer.add_special_tokens(special_tokens)
    data_collator = LLMDataCollator(tokenizer=tokenizer)



    for client, dataset in zip(global_vars.Client_list, datasets):
        loader= DataLoader(dataset, shuffle=True, batch_size=args.batch_size, collate_fn=data_collator)
        client.setup_trainloader(loader)
        client.setup_device(pidx)
    
    if pidx == 0:
        pass
def worker_local_step():
    for client in global_vars.Client_list:
        client.local_iteration()
        
def worker_communicate():
    for client in global_vars.Client_list:
        client.communicate()


def worker_gossip_result_update():
    for client in global_vars.Client_list:
        num_epoch, num_batch, total_comm = client.gossip_result_update()
    return num_epoch, num_batch, total_comm

def worker_upload_model():
    for client in global_vars.Client_list:
        client.update_model_dict()

def eval_avg_model(seed_pool, eval_matric, cur_round):
    dosomething = 0