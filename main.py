import argparse
import os
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
import torch.distributed.rpc as rpc
import torch.multiprocessing as mp

from util.arguments import get_args




def main_():
    dummy = 0
    
    ## 1. load model
    
    ## 2. load dataset
    
    ## 


def run_worker(rank, world_size, args):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = args.master_port

    if rank == 0:
        rpc.init_rpc("master", rank=rank, world_size=world_size)
        main_(world_size, args)
    else:
        i = rank-1
        rpc.init_rpc(f"gpu_{i}", rank=rank, world_size=world_size)
        # parameter server do nothing
        pass

    # block until all rpcs finish
    rpc.shutdown()



if __name__ == '__main__':
    args = get_args()
    world_size = 1 + len(args.gpu_ids)
    print(world_size)
    mp.spawn(run_worker, args=(world_size,args ), nprocs=world_size, join=True)
