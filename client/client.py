from util.topology import generate_P
from copy import deepcopy
import sys
import wandb

import torch 
import time
import copy
from torch.optim.lr_scheduler import MultiStepLR, CosineAnnealingLR, StepLR
import logging

import torch.nn as nn
import torch.optim as optim

logging.basicConfig(filename='remote_worker.log', level=logging.DEBUG)

#from util.randpro import RandProReducer
#from scheduler import Warmup_LamdbaLR, Warmup_MultiStepLR

def pool_plus(pool1:dict, pool2:dict):
    seed_pool = {}
    for seed, grad in pool1.items():
        seed_pool[seed] = grad

    for seed, grad in pool2.items():
        if seed not in seed_pool:
            seed_pool[seed] = grad
        else:
             seed_pool[seed] += grad
    return seed_pool


def pool_minus(pool1:dict, pool2:dict):
    #pool1 - pool2
    seed_pool = {}
    for seed, grad in pool1.items():
        seed_pool[seed] = grad

    for seed, grad in pool2.items():
        if seed not in seed_pool:
            seed_pool[seed] = -1 * grad
        else:
            seed_pool[seed] -= grad
    return seed_pool

def pool_scalar_product(pool1:dict, weight):
    #pool1 - pool2
    seed_pool = {}
    for seed, grad in pool1.items():
        seed_pool[seed] = grad * weight
    return seed_pool


def pool_check(pool1:dict):
    for seed, grad in pool1.items():
        assert grad.dim() != 0



class Client():
    def __init__(self, args, model, idx, candidate_seeds, device, comm_rref):
        self.args = args
        self.model = model
        self.idx = idx
        self.candidate_seeds = candidate_seeds
    
        self.device = torch.device(f'cuda:{device}')
        self.total_batch = 0
        self.total_epoch = 0
        self.total_round = 0
        self.total_communication = 0         
        self.comm_rref = comm_rref
        self.opt_strategy = args.opt_strategy
        
        if self.args.opt_strategy == "DSGD":
            self.criterion = nn.BCELoss()
            self.optimizer = optim.SGD(self.model.parameters(), lr=0.2 ,weight_decay= 0.001)

        
    def setup_trainloader(self, train_loader):
        self.train_loader = train_loader
        self.train_iterator = iter(self.train_loader)

    def local_iteration(self):
        self.total_round += 1
        if self.args.opt_strategy == "DSGD":
            self.DSGD_local_lter()
            self.DSGD_communicate()    
    def DSGD_local_lter(self):
        
        self.model = self.model.to(self.device)
        try:
            batch = next(self.train_iterator)
            self.total_batch +=1
        except StopIteration:
            self.train_iterator = iter(self.train_loader)
            self.total_epoch +=1
            batch = next(self.train_iterator)
            self.total_batch +=1
        data, target = batch[0].to(self.device).float(), batch[1].to(self.device).float()
        self.optimizer.zero_grad()
        outputs = self.model(data)
        loss = self.criterion(outputs.float(), target.float())
        loss.backward()
        self.optimizer.step()
        
        if self.idx == 0:
            logging.info(f"client idx : {self.idx} loss : {loss.item()} total batch {self.total_batch}, total epoch :{self.total_epoch} total_comm : {self.total_communication}")

            

        pass
    def ZO_local_lter(self):
        
        
        pass
    def CHOCO_local_lter(self):
        pass
        
    def DSGD_communicate(self):
        self.comm_rref.rpc_sync().update_data(self.idx, self.model.cpu().state_dict())
        self.total_communication += sum(param.numel() * param.element_size() for param in self.model.parameters())


    def gossip_result_update(self):
        with torch.no_grad():
            gossip_result = self.comm_rref.rpc_sync().get_data(self.idx)
            if self.args.opt_strategy == "DSGD":
                for name, param in self.model.named_parameters():
                    param.copy_(gossip_result[name])
        return self.total_epoch, self.total_batch, self.total_communication
    
class BasicCommunicator(object):
    def __init__(self, args):
        self.m = generate_P(args.topology, args.num_clients)
        self.data = []
    def make_avg_model(self):
        pass
    
    def all_reduce(self):
        pass
    
    def update_data(self, idx, data):
        pass
    
    def get_data(self, idx):
        pass
    
class DSGD_Communicator(object):
    def __init__(self, args):
        self.m = generate_P(args.topology, args.num_clients)
        self.args = args
        self.data = [{} for _ in range(args.num_clients)]
    def make_avg_model(self):
        avg_state_dict = copy.deepcopy(self.data[0])
        for key in avg_state_dict.keys():
            avg_state_dict[key].zero_()
        for state_dict in self.data:
            for key in state_dict.keys():
                avg_state_dict[key] += state_dict[key]
        
        for key in avg_state_dict.keys():
            avg_state_dict[key] /= self.args.num_clients
        return avg_state_dict
        
    def all_reduce(self):
        new_dict_list = copy.deepcopy(self.data)
        for d in new_dict_list:
            for key in d.keys():
                d[key].zero_()

        for i in range(self.args.num_clients):
            for j in range(self.args.num_clients):
                for key in new_dict_list[i].keys():
                    new_dict_list[i][key] += self.data[j][key] * self.m[j][i]
        
        self.data = new_dict_list
    
    def update_data(self, idx, data):
        self.data[idx] = data

    def get_data(self, idx):
        return self.data[idx]



class Evaluator(object):
    def __init__(self, args, candidate_seeds, device):
        self.args = args
        self.eval_loader = None
        self.eval_iterator = None
        self.model = None


