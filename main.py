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
import pickle
import copy
import logging
import torch.nn as nn

from torch.utils.data import DataLoader, random_split, TensorDataset

from client.client import Client, DSGD_Communicator


from util.arguments import get_args
from util.setup_seed import setup_seed
from model.logistic_regression import LogisticRegression
from torchvision import datasets, transforms

logging.basicConfig(filename='remote_worker.log', level=logging.DEBUG)

def main_(world_size, args):
    dummy = 0
    
    model = LogisticRegression(784)
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)

    def binary_classification_label_transform(label):
        return 0.0 if label < 5 else 1.0
    train_dataset.targets = torch.tensor([binary_classification_label_transform(label) for label in train_dataset.targets])
    
    
    
    dataset = train_dataset
    device = torch.device(f'cuda:0')

    n = args.num_clients
    batch_size = args.batch_size
    lengths = [len(dataset) // n] * n  # 
    lengths[-1] += len(dataset) % n  # 
    generator = torch.Generator().manual_seed(42)
    subsets = random_split(dataset, lengths, generator=generator)
    dataloaders = [DataLoader(subset, batch_size=batch_size, shuffle=False) for subset in subsets]
    train_dataloader = DataLoader(dataset, batch_size=60000, shuffle=False)
    communicator = DSGD_Communicator(args)
    comm_rref = rpc.RRef(communicator)

    ## client list에 client 들 생성
    client_list = []
    for i in range(args.num_clients):
        client_list.append(Client(args, model, i, None, 0, comm_rref))
    
    for client, dataloader in zip(client_list, dataloaders):
        client.setup_trainloader(dataloader)
        
        
    criterion = nn.BCELoss()  # 이진 교차 엔트로피 손실 함수
    ## 각 client에 dataloader 할당.
    ## init train
    num_epoch = 0
    num_batch = 0
    total_comm = 0
    while(num_batch < 100):
        for client in client_list:
            client.local_iteration()
        communicator.all_reduce()
        for client in client_list:
            num_epoch, num_batch, total_comm = client.gossip_result_update()
    
        if num_batch % 5 == 0 :
            with torch.no_grad():
                avg_model_dict = communicator.make_avg_model()
                avgmodel = copy.deepcopy(model)
                avgmodel = avgmodel.to(device)
                for name, param in avgmodel.named_parameters():
                    param.copy_(avg_model_dict[name])
                total_loss = 0.0
                for A, y in train_dataloader:
                    A, y = A.to(device).float(), y.to(device).float()
                    outputs = avgmodel(A)  # 전체 데이터를 모델에 입력
                    loss = criterion(outputs, y)  # 손실 계산
                    total_loss += loss.item()
                logging.info(f"avg model loss : {total_loss/len(train_dataloader)}, at epoch{num_epoch} iter {num_batch}")
                

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
    world_size = 1 + len(args.gpu_ids) if len(args.gpu_ids) > 1 else 1
    if len(args.gpu_ids) > 1:
        world_size = 1 + len(args.gpu_ids)
    else:
        world_size = 1
    mp.spawn(run_worker, args=(world_size,args ), nprocs=world_size, join=True)
