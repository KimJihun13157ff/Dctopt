import os
import time
import torch
import copy
import random
import datetime
import argparse



# get ArgumentParser
def get_args():
    parser = argparse.ArgumentParser()
    
    # multi gpu
    parser.add_argument('--gpu_ids', nargs="+", default=['0', '1', '2', '3'])# --gpu_ids 0 1 2 3
    parser.add_argument('--master_port', type=str, default='29400')
    parser.add_argument("--project_name", type=str, default='dct_09_19')
    
    parser.add_argument('--seed', type=int, default=777)

    parser.add_argument("--batch_size", type=int, default=64)

    parser.add_argument("--opt_strategy", type=str, default='DSGD',
                                            choices=['DSGD','DZOK','CHOCO', 'DZO', 'CHOCO_DZO', 'DZOPK', 'FLOOD'])
    parser.add_argument("--topology", type=str, default='ring',
                                            choices=['ring','meshgrid','all'])
    
    parser.add_argument('--num_clients', type=int, default=12)
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
    
    parser.add_argument('--weight_decay', type=float, default=0.001, help='weight decay in MeZO')
    parser.add_argument('--lr_decay', type=float, default=0.998)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--communicate_round', type=int, default=500)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--optimization", type=str, default='sgd')
    parser.add_argument('--warmup_step', type=int, default=5)
    parser.add_argument('--grad_clip', type=float, default=-100.0, help='clip the over large loss value, if < 0, disable this feature')
    parser.add_argument('--zo_normalized', action='store_true', help='use ZO normalization if set')

    parser.add_argument('--max_epoch', type=int, default=300)
    parser.add_argument('--K', type=int, default=4096, help='')
    parser.add_argument('--Period', type=int, default=100, help='')
    parser.add_argument('--npert', type=int, default=1, help='')


    parser.add_argument("--gossip_iter", type=int, default=1)
    parser.add_argument("--local_iter", type=int, default=1)
    
    
    parser.add_argument('--zo_eps', type=float, default=0.0005, help=r'\eps in MeZO')
    parser.add_argument('--consensus_ss', type=float, default=0.2, help="consensus step size for compressed gossip")
    parser.add_argument('--sparsification_r', type=float, default=0.9, help="dropout rate of sparsificaiton")
    
    
    # for llm
    parser.add_argument('--dataset', type=str, default='instruct', choices=['instruct', 'dolly'])
    parser.add_argument('--max_length', type=int, default=1024, help='the max number of tokens of a data instance')
    parser.add_argument('--use_prompts', default=True, help='if `true`, the prompt template from alpaca is adopted')
    parser.add_argument('--model', type=str, default='facebook/opt-125m', choices=['facebook/opt-125m', 'datajuicer/LLaMA-1B-dj-refine-150B', "meta-llama/Llama-3.2-3B"])



    # multi gpu
    '''

    ## dataset
    parser.add_argument("--dataset_path", type=str, default='/jihun')
    parser.add_argument("--dataset_name", type=str, default='CIFAR10',
                                            choices=['CIFAR10','TinyImageNet'])
    ## dataset-for image
    parser.add_argument("--image_size", type=int, default=32, help='input image size')


    ## model
    parser.add_argument('--model', type=str, default='ResNet18', choices=['ResNet18', 'TBD'])



    ## network
    parser.add_argument("--topology", type=str, default='ring',
                                            choices=['ring','meshgrid',''])
    parser.add_argument('--num_clients', type=int, default=12)
    


    # optimization parameter


    ## optimization scheduling parameter
    parser.add_argument('--milestones', type=int, default=[150, 250])
    parser.add_argument('--rounds', type=int, default=3000)


    # Training args for limited seed`
    parser.add_argument('--zo_eps', type=float, default=0.0005, help=r'\eps in MeZO')

    '''
    ##
    args = parser.parse_args()
    return args



