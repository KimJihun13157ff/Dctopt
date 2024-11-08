import argparse
import os
import time
import random
import math
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

from utils_data.default_tokens import DefaultToken

from transformers import AutoModelForCausalLM, AutoTokenizer

from client.client import DZO_Client, DSGD_Communicator, DZOK_Communicator, DSGD_Client, DZOK_Client, CHOCO_Client, CHOCO_Communicator, CHOCO_DZO_Client
from client.Fgossip import DZO_FLOODgossip_Client, DZO_FLOODgossip_Communicator

from util.arguments import get_args
from util.setup_seed import setup_seed
from model.logistic_regression import LogisticRegression
from model.cnn_mnist import CNN_MNIST
from model.resnet_s import resnet20
from model.resnet import resnet18
from rpc_function import worker_set_up, worker_gossip_result_update, worker_local_step, worker_communicate, worker_upload_model

from torchvision import datasets, transforms

logging.basicConfig(level=logging.DEBUG)

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def main_(world_size, args):

    setup_seed(args.seed)
    if args.opt_strategy == 'DZOK'or args.opt_strategy == 'DZOPK':
        additional = args.K
    elif args.opt_strategy == 'CHOCO' or args.opt_strategy == 'CHOCO_DZO':
        additional = args.sparsification_r
    else:
        additional = ""

    wandb.init(project=f"{args.project_name}",
    name=f'{args.opt_strategy}_seed{args.seed}_lr{args.lr}_{additional}_css{args.consensus_ss}_{args.local_iter}to{args.gossip_iter}',
    config={
        "args" : args
    })
    logging.info(args)
    print(args)

    
    # load model.    
    model = AutoModelForCausalLM.from_pretrained(args.model, device_map='cpu', torch_dtype=torch.float16, trust_remote_code=True)
    '''
    if args.model == "logistic_regression":
        model = LogisticRegression(784)
    elif args.model == "CNN_MNIST":
        model = CNN_MNIST()
    elif args.model == "Resnet20":
        model = resnet20()
    elif args.model == "Resnet18":
        model = resnet18()
        model.fc = nn.Linear(model.fc.in_features, 100)
    '''
    #candidate_seeds = np.random.randint(1, 2147483647, args.K)


    candidate_seeds = torch.randint(low=1, high=2147483647, size=(args.K,), dtype=torch.int32)
    # load dataset
    
    list_train_dataset, eval_dataset, tokenizer = get_loaders(args)
    
    '''
    if args.model == "CNN_MNIST":
        transform = transforms.Compose([transforms.ToTensor()])
        train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        validation_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    elif args.model == "Resnet20":
        mean = [0.4914, 0.4822, 0.4465]
        std = [0.2471, 0.2435, 0.2616]
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        train_dataset = datasets.CIFAR10(root ='./data', train = True, download = True, transform = train_transform)
        validation_dataset = datasets.CIFAR10(root = './data', train = False, download = True, transform = test_transform)
    elif args.model == "Resnet18":
        train_data = datasets.CIFAR100('./data', train=True, download=True)

        # Stick all the images together to form a 1600000 X 32 X 3 array
        x = np.concatenate([np.asarray(train_data[i][0]) for i in range(len(train_data))])

        # calculate the mean and std along the (0, 1) axes
        mean = np.mean(x, axis=(0, 1))/255
        std = np.std(x, axis=(0, 1))/255
        # the the mean and std
        mean=mean.tolist()
        std=std.tolist()
        train_transform = transforms.Compose([transforms.RandomCrop(32, padding=4,padding_mode='reflect'), 
                                transforms.RandomHorizontalFlip(), 
                                transforms.ToTensor(), 
                                transforms.Normalize(mean,std,inplace=True)])
        test_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean,std)])

        train_dataset = datasets.CIFAR100(root ='./data', train = True, download = True, transform = train_transform)
        validation_dataset = datasets.CIFAR100(root = './data', train = False, download = True, transform = test_transform)
        ## https://www.kaggle.com/code/yiweiwangau/cifar-100-resnet-pytorch-75-17-accuracy
    
    
    
    def binary_classification_label_transform(label):
        return 0 if label < 5 else 1
    
    
    if args.model == "logistic_regression":
        train_dataset.targets = torch.tensor([binary_classification_label_transform(label) for label in train_dataset.targets])
        validation_dataset.targets = torch.tensor([binary_classification_label_transform(label) for label in validation_dataset.targets])
    '''
    ## make global dataloader.
    """
    validation_dataloader = DataLoader(validation_dataset, batch_size=128, shuffle=False)
    """
    data_collator = LLMDataCollator(tokenizer=tokenizer)
    validation_dataloader= DataLoader(eval_dataset, batch_size=args.batch_size, collate_fn=data_collator)
    subsets = list_train_dataset
    ### communicator
    if args.opt_strategy == 'DSGD' or args.opt_strategy == "DZO":
        communicator = DSGD_Communicator(args)
    elif args.opt_strategy == 'DZOK' or args.opt_strategy == 'DZOPK':
        communicator = DZOK_Communicator(args, model, candidate_seeds)
    elif args.opt_strategy == 'CHOCO' or args.opt_strategy == 'CHOCO_DZO':
        communicator = CHOCO_Communicator(args, args.consensus_ss)
    elif args.opt_strategy == 'FLOOD':
        communicator = DZO_FLOODgossip_Communicator(args)
    comm_rref = rpc.RRef(communicator)

## client
    client_list = []
    if args.opt_strategy == 'DSGD':
        for i in range(args.num_clients):
            client_list.append(DSGD_Client(args, model, i, 0, comm_rref))
    if args.opt_strategy == 'DZO':
        for i in range(args.num_clients):
            client_list.append(DZO_Client(args, model, i, 0, comm_rref))
    if args.opt_strategy == 'DZOK' or args.opt_strategy == 'DZOPK' :
        for i in range(args.num_clients):
            client_list.append(DZOK_Client(args, model, i, 0, comm_rref,candidate_seeds))
    if args.opt_strategy == 'CHOCO':
        for i in range(args.num_clients):
            client_list.append(CHOCO_Client(args, model, i, 0, comm_rref,args.sparsification_r))
    if args.opt_strategy == 'CHOCO_DZO':
        for i in range(args.num_clients):
            client_list.append(CHOCO_DZO_Client(args, model, i, 0, comm_rref,args.sparsification_r))
    if args.opt_strategy == 'FLOOD':
        for i in range(args.num_clients):
            client_list.append(DZO_FLOODgossip_Client(args, model, i, 0, comm_rref))

    if world_size == 1:
        dataloaders = [DataLoader(subset, batch_size=batch_size, shuffle=False) for subset in subsets]
        for client, dataloader in zip(client_list, dataloaders):
            client.setup_trainloader(dataloader)
    else: # client setup ## 각 client에 dataloader 할당.

        futures =[]
        n_worker = world_size - 1
        assert args.num_clients % n_worker == 0
        for i in range(n_worker):
            start = int(args.num_clients / n_worker)* i
            end =  int(args.num_clients / n_worker)  * (i + 1)
            futures.append(rpc.rpc_async(f"gpu_{i}", worker_set_up ,args=(args, i, client_list[start:end], subsets[start: end] )))
        [fut.wait() for fut in futures]
    
    ## init train
    num_epoch = 0
    num_batch = 0
    total_comm = 0
    while(num_batch < args.max_epoch):
        ## train
        
        # local step
        if world_size == 1:
            for client in client_list:
                client.local_iteration()
        else:
            futures = []
            for i in range(n_worker):
                futures.append(rpc.rpc_async(f"gpu_{i}", worker_local_step))
            [fut.wait() for fut in futures]
        # communication step
        for _ in range(args.gossip_iter):
            ## send model to communicator
            if world_size == 1:
                for client in client_list:
                    client.communicate()
            else:
                futures = []
                for i in range(n_worker):
                    futures.append(rpc.rpc_async(f"gpu_{i}", worker_communicate))
                [fut.wait() for fut in futures]
            ## model all reduce
            communicator.all_reduce()
            ## recieve communicate  result and update model
            if world_size == 1:
                for client in client_list:
                    num_epoch, num_batch, total_comm = client.gossip_result_update()
            else:
                futures = []
                for i in range(n_worker):
                    futures.append(rpc.rpc_async(f"gpu_{i}", worker_gossip_result_update))
                result = [fut.wait() for fut in futures]
                num_epoch, num_batch, total_comm = result[0]
                
        if args.opt_strategy == "FLOOD":
            total_comm = communicator.get_communication()        
        ## validation    
        if num_batch % 50 == 0 :
            
            if args.opt_strategy == "FLOOD" or args.opt_strategy == "DZOK":
                if world_size == 1:
                    for client in client_list:
                        client.upload_model()
                else:
                    futures = []
                    for i in range(n_worker):
                        futures.append(rpc.rpc_async(f"gpu_{i}", worker_upload_model))
                    [fut.wait() for fut in futures]

            avg_model_dict = communicator.make_avg_model()
            avgmodel = copy.deepcopy(model)
            avgmodel = avgmodel.to(device)
            for name, param in avgmodel.named_parameters():
                with torch.no_grad():
                    param.copy_(avg_model_dict[name])
            total_train_loss = 0.0
            with torch.no_grad():
                correct = 0
                total = 0
                for A, y in train_dataloader:
                    A, y = A.to(device), y.to(device)
                    outputs = avgmodel(A)  # 전체 데이터를 모델에 입력
                    loss = criterion(outputs, y)  # 손실 계산
                    total_train_loss += loss.item()
                    _, predicted = torch.max(outputs, 1)
                    total += y.size(0)
                    correct += (predicted == y).sum().item()
                val_correct = 0
                val_total = 0
                total_validation_loss = 0
                
                for A, y in validation_dataloader:
                    A, y = A.to(device), y.to(device)
                    outputs = avgmodel(A)
                    loss = criterion(outputs, y)
                    total_validation_loss += loss.item()
                    _, predicted = torch.max(outputs, 1)
                    val_total += y.size(0)
                    val_correct += (predicted == y).sum().item()
                    
                consensus_distance = communicator.consensus_distance()            
                wandb.log({"Avgmodel Training loss": total_train_loss/len(train_dataloader), "consensus distance": consensus_distance, "total_comm":int(total_comm), "iteration":num_batch, "Avgmodel Validation loss":total_validation_loss/len(validation_dataloader), "train_accuracy":correct/total, "validation_accuracy":val_correct/val_total})
                logging.info(f"avg model loss : {total_train_loss/len(train_dataloader)}, at epoch {num_epoch} iter {num_batch}")
    
                if (math.isnan(total_train_loss)):
                    return 
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
