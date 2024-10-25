from util.topology import generate_P
from mezo_framework.mezo_optimizer import MeZOFramework
from compression.sparsification import Sparsification


from copy import deepcopy
import sys
import wandb

import torch 
import time
import copy
from torch.optim.lr_scheduler import MultiStepLR, CosineAnnealingLR, StepLR
import logging
import numpy as np

import torch.nn as nn
import torch.optim as optim

logging.basicConfig(level=logging.DEBUG)

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
    def __init__(self, args, model, idx, device, comm_rref):
        self.args = args
        self.model = copy.deepcopy(model)
        self.idx = idx    
        self.device = torch.device(f'cuda:{device}')
        self.total_batch = 0
        self.total_epoch = 0
        self.total_round = 0
        self.total_communication = 0         
        self.comm_rref = comm_rref
        self.opt_strategy = args.opt_strategy
    def setup_trainloader(self, train_loader):
        self.train_loader = train_loader
        self.train_iterator = iter(self.train_loader)
        
    def local_iteration(self):
        pass
    def gossip_result_update(self):
        pass
    def communicate(self):
        pass

class dummy_Client(Client):
    def __init__(self, args, model, idx, device, comm_rref):
        super().__init__(args, model, idx, device, comm_rref)
    def local_iteration(self):
        pass
    def gossip_result_update(self):
        pass
    

class CHOCO_Client(Client):
    def __init__(self, args, model, idx, device, comm_rref, sparsificaition_r):
        super().__init__(args, model, idx, device, comm_rref)
        if args.model == "logistic_regression":
            self.criterion = nn.BCELoss()
        else:
            self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.args.lr ,weight_decay= self.args.weight_decay)
        self.x_hat = {name: torch.zeros_like(param).to(self.device) for name, param in model.named_parameters()}
        self.compressor = Sparsification(op= 'top_k', ratio=sparsificaition_r) # 1-r% are alive
        
        
    def local_iteration(self):
        self.total_round += 1
        self.DSGD_local_iter()
    def communicate(self):
        with torch.no_grad():
            self.CHOCO_communicate()


    def gossip_result_update(self):
        with torch.no_grad():
            gossip_result = self.comm_rref.rpc_sync().get_data(self.idx)
            for name, param in self.model.named_parameters():
                param.copy_(gossip_result[name])
        return self.total_epoch, self.total_batch, self.total_communication
      
        
    
    def CHOCO_communicate(self):
        model_dict = {}
        for name, x in self.model.named_parameters():
            model_dict[name], mask=  self.compressor.get_top_k(x.to(self.device) - self.x_hat[name].to(self.device), ratio=self.compressor.ratio)
            self.total_communication += x.element_size() * x.nelement() * (1 - self.compressor.ratio) * 2 # (2 mean coordinate + value)
            # update x_hat
            self.x_hat[name] += model_dict[name]
            model_dict[name]= model_dict[name].cpu()
        self.comm_rref.rpc_sync().update_data(self.idx, {name: param for name, param in self.model.cpu().named_parameters()} ,model_dict)

        #1.calculate x-xhat and compress 
    
    def DSGD_local_iter(self):
        
        self.model = self.model.to(self.device)
        for _ in range(self.args.local_iter):
            try:
                batch = next(self.train_iterator)
                self.total_batch +=1
            except StopIteration:
                self.train_iterator = iter(self.train_loader)
                self.total_epoch +=1
                batch = next(self.train_iterator)
                self.total_batch +=1
            data, target = batch[0].to(self.device).float(), batch[1].to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(data)
            loss = self.criterion(outputs, target)
            loss.backward()
            self.optimizer.step()
        
        
        if self.idx == 0:
            logging.info(f"client idx : {self.idx} loss : {loss.item()} total batch {self.total_batch}, total epoch :{self.total_epoch} total_comm : {self.total_communication}")        


class CHOCO_DZO_Client(Client):
    def __init__(self, args, model, idx, device, comm_rref, sparsificaition_r):
        super().__init__(args, model, idx, device, comm_rref)
        self.criterion = nn.BCELoss() if args.model == "logistic_regression" else nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.args.lr ,weight_decay= self.args.weight_decay)
        self.x_hat = {name: torch.zeros_like(param).to(self.device) for name, param in model.named_parameters()}
        self.compressor = Sparsification(op= 'top_k', ratio=sparsificaition_r) # 1-r% are alive
        
        
    def local_iteration(self):
        self.total_round += 1
        self.DZO_local_iter()
    def communicate(self):
        with torch.no_grad():
            self.CHOCO_communicate()
    
    def gossip_result_update(self):
        with torch.no_grad():
            gossip_result = self.comm_rref.rpc_sync().get_data(self.idx)
            for name, param in self.model.named_parameters():
                param.copy_(gossip_result[name])
        return self.total_epoch, self.total_batch, self.total_communication
      
        
    
    def CHOCO_communicate(self):
        model_dict = {}
        for name, x in self.model.named_parameters():
            model_dict[name], mask=  self.compressor.get_top_k(x.to(self.device) - self.x_hat[name].to(self.device), ratio=self.compressor.ratio)
            self.total_communication += x.element_size() * x.nelement() * (1 - self.compressor.ratio) * 2 # (2 mean coordinate + value)
            # update x_hat
            self.x_hat[name] += model_dict[name]
            model_dict[name]= model_dict[name].cpu()
        self.comm_rref.rpc_sync().update_data(self.idx, {name: param for name, param in self.model.cpu().named_parameters()} ,model_dict)

        #1.calculate x-xhat and compress 
    
    def DZO_local_iter(self):
        
        lr = self.args.lr
        framework = MeZOFramework(self.model, args=self.args, lr=lr, candidate_seeds=None)
        self.model.eval()
        self.model = self.model.to(self.device)
        loss_total_train  = 0
        for _ in range(self.args.local_iter):
            try:
                batch = next(self.train_iterator)
                self.total_batch +=1
            except StopIteration:
                self.train_iterator = iter(self.train_loader)
                self.total_epoch +=1
                batch = next(self.train_iterator)
                self.total_batch +=1
            logits, loss = framework.zo_step(batch)
            if (not torch.isnan(loss)) and (loss != 0.0):
                loss_total_train += loss
        if self.idx == 0:
            logging.info(f"client idx : {self.idx} loss : {loss} total batch {self.total_batch}, total epoch :{self.total_epoch} total_comm : {self.total_communication}")





    
class DSGD_Client(Client):
    def __init__(self, args, model, idx, device, comm_rref):
        super().__init__(args, model, idx, device, comm_rref)
        if args.model == "logistic_regression":
            self.criterion = nn.BCELoss()
        else:
            self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.args.lr ,weight_decay= self.args.weight_decay)
    def local_iteration(self):
        self.total_round += 1
        self.DSGD_local_iter()
    def communicate(self):
        self.DSGD_communicate()

    def DSGD_local_iter(self):
        
        self.model = self.model.to(self.device)
        self.model.train()
        for _ in range(self.args.local_iter):
            try:
                batch = next(self.train_iterator)
                self.total_batch +=1
            except StopIteration:
                self.train_iterator = iter(self.train_loader)
                self.total_epoch +=1
                batch = next(self.train_iterator)
                self.total_batch +=1
            data, target = batch[0].to(self.device).float(), batch[1].to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(data)
            loss = self.criterion(outputs, target)
            loss.backward()
            self.optimizer.step()
        
        if self.idx == 0:
            logging.info(f"client idx : {self.idx} loss : {loss.item()} total batch {self.total_batch}, total epoch :{self.total_epoch} total_comm : {self.total_communication}")        
            
    def DSGD_communicate(self):
        self.model.eval()
        self.comm_rref.rpc_sync().update_data(self.idx, {name: param for name, param in self.model.cpu().named_parameters()})
        self.total_communication += sum(param.numel() * param.element_size() for param in self.model.parameters())

    def gossip_result_update(self):
        with torch.no_grad():
            gossip_result = self.comm_rref.rpc_sync().get_data(self.idx)
            for name, param in self.model.named_parameters():
                    param.copy_(gossip_result[name])
        return self.total_epoch, self.total_batch, self.total_communication


class DZO_Client(Client):
    def __init__(self, args, model, idx, device, comm_rref):
        super().__init__(args, model, idx, device, comm_rref)

    def local_iteration(self):
        self.total_round += 1
        self.DZO_local_iter()
    def communicate(self):
        self.DZO_communicate()
        
    def DZO_local_iter(self):
        
        lr = self.args.lr
        framework = MeZOFramework(self.model, args=self.args, lr=lr, candidate_seeds=None)
        self.model.eval()
        self.model = self.model.to(self.device)
        loss_total_train  = 0
        for _ in range(self.args.local_iter):
            try:
                batch = next(self.train_iterator)
                self.total_batch +=1
            except StopIteration:
                self.train_iterator = iter(self.train_loader)
                self.total_epoch +=1
                batch = next(self.train_iterator)
                self.total_batch +=1
            logits, loss = framework.zo_step(batch)
            if (not torch.isnan(loss)) and (loss != 0.0):
                loss_total_train += loss
        #wandb.log({f"model{self.idx} Training loss": loss_total_train/self.args.local_iter})
        if self.idx == 0:
            logging.info(f"client idx : {self.idx} loss : {loss} total batch {self.total_batch}, total epoch :{self.total_epoch} total_comm : {self.total_communication}")

    def DZO_communicate(self):
        self.comm_rref.rpc_sync().update_data(self.idx, {name: param for name, param in self.model.cpu().named_parameters()})
        self.total_communication += sum(param.numel() * param.element_size() for param in self.model.parameters())
    def gossip_result_update(self):
        with torch.no_grad():
            gossip_result = self.comm_rref.rpc_sync().get_data(self.idx)
            for name, param in self.model.named_parameters():
                param.copy_(gossip_result[name])
        return self.total_epoch, self.total_batch, self.total_communication



class DZOK_Client(Client):
    def __init__(self, args, model, idx, device, comm_rref, candidate_seeds):
        super().__init__(args, model, idx, device, comm_rref)
        self.candidate_seeds = candidate_seeds
        self.total_seed_pool = {int(seed) :torch.tensor(0.0, dtype=torch.float16) for seed in self.candidate_seeds}
        self.pt_model = model
        
    def local_iteration(self):
        self.total_round += 1
        self.DZOK_local_iter()
    def communicate(self):
        self.DZOK_communicate()
    def DZOK_local_iter(self):
        
        lr = self.args.lr
        framework = MeZOFramework(self.model, args=self.args, lr=lr, candidate_seeds=self.candidate_seeds)
        self.model.eval()
        self.model = self.model.to(self.device)
        loss_total_train  = 0
        
        for _ in range(self.args.local_iter):
            try:
                batch = next(self.train_iterator)
                self.total_batch +=1
            except StopIteration:
                self.train_iterator = iter(self.train_loader)
                self.total_epoch +=1
                batch = next(self.train_iterator)
                self.total_batch +=1
            logits, loss = framework.zo_step(batch, local_seed_pool=self.total_seed_pool)
        
        #if (not torch.isnan(loss)) and (loss != 0.0):
            #loss_total_train += loss
        if self.idx == 0:
            logging.info(f"client idx : {self.idx} loss : {loss} total batch {self.total_batch}, total epoch :{self.total_epoch} total_comm : {self.total_communication}")
        
    def DZOK_communicate(self):
        def get_dict_size(d):
            total_size = 0  
            for key, value in d.items():
                if value != 0.0:
                    total_size += torch.tensor(key, dtype=torch.int32).element_size()   # 각 키의 메모리 크기 추가
                    total_size += value.element_size()  # 각 값의 메모리 크기 추가
            return total_size
        self.comm_rref.rpc_sync().update_data(self.idx, self.total_seed_pool, {name: param for name, param in self.model.cpu().named_parameters()})
        self.total_communication += get_dict_size(self.total_seed_pool)
    def gossip_result_update(self):
        self.total_seed_pool = self.comm_rref.rpc_sync().get_data(self.idx)
        self.model = copy.deepcopy(self.pt_model)
        self.model.to(self.device)
        framework = MeZOFramework(self.model, args=self.args, lr=self.args.lr, candidate_seeds=self.candidate_seeds)
        for seed, grad in self.total_seed_pool.items():
            if grad != 0.0:
                framework.zo_update(seed=seed, grad=grad)
        del framework
        return self.total_epoch, self.total_batch, self.total_communication
    
    def DZOK_MGS_comm_update(self): # this function is for multiple gossip step. only update comm cost
        self.total_seed_pool = self.comm_rref.rpc_sync().get_data(self.idx)
        def get_dict_size(d):
            total_size = 0  
            for key, value in d.items():
                if value != 0.0:
                    total_size += torch.tensor(key, dtype=torch.int32).element_size()   # 각 키의 메모리 크기 추가
                    total_size += value.element_size()  # 각 값의 메모리 크기 추가
            return total_size
        self.total_communication += get_dict_size(self.total_seed_pool)
        return self.total_epoch, self.total_batch, self.total_communication
    
    def DZOK_new_init_model(self, candidated_seed):
        self.pt_model = copy.deepcopy(self.model)
        self.candidate_seeds = candidated_seed
        self.total_seed_pool = {int(seed) :torch.tensor(0.0, dtype=torch.float16) for seed in self.candidate_seeds}


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
    
    
class DZOK_Communicator(object):
    def __init__(self, args, init_model, candidate_seeds):
        self.m = generate_P(args.topology, args.num_clients)
        self.args = args
        self.data = [{} for _ in range(args.num_clients)]
        self.init_model = init_model
        self.candidate_seeds = candidate_seeds
        
        self.model_dict = [{} for _ in range(args.num_clients)]
    
    def consensus_distance(self):
        avg_model_dict = self.make_avg_model()
        consensus_distance = 0
        for i in range(self.args.num_clients):
            for key in avg_model_dict.keys():
                consensus_distance += torch.norm(avg_model_dict[key] - self.model_dict[i][key]) ** 2
        consensus_distance /= self.args.num_clients        
        return consensus_distance ** 1/2
    
    def make_avg_model(self):
        avg_state_dict = {key: torch.zeros_like(tensor) for key, tensor in self.data[0].items()}
            
        for state_dict in self.model_dict:
            for key in state_dict.keys():
                avg_state_dict[key] += state_dict[key]
        
        for key in avg_state_dict.keys():
            avg_state_dict[key] /= self.args.num_clients
        return avg_state_dict
        
    '''
    # note : something wrong in this avg model making source...        
    def make_avg_model(self):
        avg_state_dict= {key: torch.tensor(0.0, dtype=torch.float16) for key in self.data[0].keys()}
        ## make avg key list
        for state_dict in self.data:
            for key in state_dict.keys():
                avg_state_dict[key] += (state_dict[key] / self.args.num_clients)
        ### make avg model
        
        model = copy.deepcopy(self.init_model)
        framework = MeZOFramework(model, args=self.args, lr=self.args.lr, candidate_seeds=self.candidate_seeds)
        for seed, grad in avg_state_dict.items():
            if grad != 0.0:
                framework.zo_update(seed=seed, grad=grad)
        del framework
        return model
    '''
    def all_reduce(self):
        new_dict_list = [{key: torch.tensor(0.0, dtype=torch.float16) for key in d.keys()} for d in self.data]
        for i in range(self.args.num_clients):
            for j in range(self.args.num_clients):
                for key in new_dict_list[i].keys():
                    new_dict_list[i][key] += self.data[j][key] * self.m[j][i]
        self.data = new_dict_list
            
    def update_data(self, idx, data, model_dict):
        self.data[idx] = data
        self.model_dict[idx] = model_dict
    
    def get_data(self, idx):
        return self.data[idx]
    

class DSGD_Communicator(object):
    def __init__(self, args):
        self.m = generate_P(args.topology, args.num_clients)
        self.args = args
        self.data = [{} for _ in range(args.num_clients)]
        

    def consensus_distance(self):
        avg_model_dict = self.make_avg_model()
        consensus_distance = 0
        for i in range(self.args.num_clients):
            for key in avg_model_dict.keys():
                consensus_distance += torch.norm(avg_model_dict[key] - self.data[i][key]) ** 2
        consensus_distance /= self.args.num_clients        
        return consensus_distance ** 1/2
    
    def make_avg_model(self):
        avg_state_dict = {key: torch.zeros_like(tensor) for key, tensor in self.data[0].items()}
        for state_dict in self.data:
            for key in state_dict.keys():
                avg_state_dict[key] += state_dict[key]
        
        for key in avg_state_dict.keys():
            avg_state_dict[key] /= self.args.num_clients
        return avg_state_dict
        
    def all_reduce(self):
        new_dict_list = [{key: torch.zeros_like(value) for key, value in d.items()} for d in self.data]
        for i in range(self.args.num_clients):
            for j in range(self.args.num_clients):
                for key in new_dict_list[i].keys():
                    new_dict_list[i][key] += self.data[j][key] * self.m[j][i]        
        self.data = new_dict_list
    
    def update_data(self, idx, data):
        self.data[idx] = data

    def get_data(self, idx):
        return self.data[idx]    
    
    
    
class CHOCO_Communicator(object):
    def __init__(self, args, consensus_ss):
        self.m = generate_P(args.topology, args.num_clients)
        self.args = args
        self.consensus_ss = consensus_ss
        self.x = [{} for _ in range(args.num_clients)]
        self.x_hat = [{} for _ in range(args.num_clients)]


    def consensus_distance(self):
        avg_model_dict = self.make_avg_model()
        consensus_distance = 0
        for i in range(self.args.num_clients):
            for key in avg_model_dict.keys():
                consensus_distance += torch.norm(avg_model_dict[key] - self.x[i][key]) ** 2
        consensus_distance /= self.args.num_clients        
        return consensus_distance ** 1/2

    def make_avg_model(self):
        avg_state_dict = {key: torch.zeros_like(tensor) for key, tensor in self.x[0].items()}
        for state_dict in self.x:
            for key in state_dict.keys():
                avg_state_dict[key] += state_dict[key]
        
        for key in avg_state_dict.keys():
            avg_state_dict[key] /= self.args.num_clients
        return avg_state_dict
        
    def all_reduce(self):
        new_dict_list = [{key: torch.zeros_like(value) for key, value in d.items()}for d in self.x]
        for i in range(self.args.num_clients):
            for j in range(self.args.num_clients):
                for key in new_dict_list[i].keys():
                    new_dict_list[i][key] += self.x_hat[j][key] * self.m[j][i]
                    

        for i in range(self.args.num_clients):
            for key in new_dict_list[i].keys():
                choco_result = self.x[i][key] + self.consensus_ss * (new_dict_list[i][key] - self.x_hat[i][key])
                new_dict_list[i][key] =choco_result

        self.x = new_dict_list
    
    def update_data(self, idx, x_updated, x_hat_delta):
        self.x[idx] = x_updated
        for key in x_hat_delta.keys():
            if key in self.x_hat[idx]:
                self.x_hat[idx][key] += x_hat_delta[key]
            else:
                self.x_hat[idx][key] = x_hat_delta[key]

    def get_data(self, idx):
        return self.x[idx]



class Evaluator(object):
    def __init__(self, args, candidate_seeds, device):
        self.args = args
        self.eval_loader = None
        self.eval_iterator = None
        self.model = None


