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

    # multi gpu
    '''

    ## dataset
    parser.add_argument("--dataset_path", type=str, default='/jihun')
    parser.add_argument("--dataset_name", type=str, default='CIFAR10',
                                            choices=['CIFAR10','TinyImageNet'])
    ## dataset-for image
    parser.add_argument("--image_size", type=int, default=32, help='input image size')
    parser.add_argument("--batch_size", type=int, default=32)


    ## model
    parser.add_argument('--model', type=str, default='ResNet18', choices=['ResNet18', 'TBD'])



    ## network
    parser.add_argument("--topology", type=str, default='ring',
                                            choices=['ring','meshgrid',''])
    parser.add_argument('--num_clients', type=int, default=12)
    


    # optimization parameter
    parser.add_argument('--optimizer_order', type=str, default='FO', choices=['FO', 'ZO'])
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='weight decay in MeZO')
    parser.add_argument('--lr_decay', type=float, default=0.998)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--communicate_round', type=int, default=500)
    parser.add_argument('--seed', type=int, default=777)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--optimization", type=str, default='sgd')
    parser.add_argument('--warmup_step', type=int, default=5)
    parser.add_argument('--grad_clip', type=float, default=-100.0, help='clip the over large loss value, if < 0, disable this feature')

    parser.add_argument("--gossip_iter", type=int, default=1)
    parser.add_argument("--local_iter", type=int, default=1)

    ## optimization scheduling parameter
    parser.add_argument('--max_epoch', type=int, default=300)
    parser.add_argument('--milestones', type=int, default=[150, 250])
    parser.add_argument('--rounds', type=int, default=3000)


    # Training args for limited seed`
    parser.add_argument('--K', type=int, default=4096, help='ratio of active clients in each round')
    parser.add_argument('--zo_eps', type=float, default=0.0005, help=r'\eps in MeZO')

    '''
    ##
    args = parser.parse_args()
    return args


