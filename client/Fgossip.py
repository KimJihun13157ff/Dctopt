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
import math


import torch.nn as nn
import torch.optim as optim
from client.client import Client

import struct


logging.basicConfig(level=logging.DEBUG)

#from util.randpro import RandProReducer
#from scheduler import Warmup_LamdbaLR, Warmup_MultiStepLR


class message(object):
    def __init__(self, data:dict, num_clients, bitmask = None):
        self.data = data
        if bitmask == None:
            self.bitmask = int('0' * num_clients, 2)
        else:
            self.bitmask = bitmask
        self.num_clients = num_clients
    def check_bit(self, client_idx):
        assert client_idx < self.num_clients
        return self.bitmask & (1<<client_idx) != 0
    
    def set_bit(self, client_idx):
        assert client_idx < self.num_clients
        self.bitmask |= (1<<client_idx)
    
    def sum_bit(self, other_bitmask):
        self.bitmask |= other_bitmask

    def pack(self):
        # 'data' 딕셔너리의 내용을 패킹
        data_items = []
        for key, value in self.data.items():
            '''
            key_bytes = key.to(torch.int32).numpy().tobytes()  # int32 형식으로 4바이트 패킹
            value_bytes = value.to(torch.float32).numpy().tobytes()  # float32 형식으로 4바이트 패킹
            data_items.append(key_bytes + value_bytes)
            '''
            key_tensor = torch.tensor(key, dtype=torch.int32).clone().detach()
            key_bytes = key_tensor.numpy().tobytes()  # int32 형식으로 4바이트 패킹
            
            
            value_tensor = torch.tensor(value, dtype=torch.float32).clone().detach()
            value_bytes = value_tensor.numpy().tobytes()  # float32 형식으로 4바이트 패킹
            data_items.append(key_bytes + value_bytes)

        packed_data = b''.join(data_items)

        # 비트맵 패킹
        num_bytes_for_bitmask = math.ceil(self.num_clients / 8)  # 비트맵에 필요한 바이트 수 계산
        packed_bitmask = self.bitmask.to_bytes(num_bytes_for_bitmask, byteorder='big')
        
        return packed_bitmask + packed_data
    
    def unpack(self, packed_bytes):
        num_bytes_for_bitmask = math.ceil(self.num_clients / 8)
        self.bitmask = int.from_bytes(packed_bytes[:num_bytes_for_bitmask], byteorder='big')

        # 데이터 언패킹
        data = {}
        data_bytes = packed_bytes[num_bytes_for_bitmask:]
        offset = 0
        data_item_size = 8  # 4바이트 int + 4바이트 float
        
        while offset < len(data_bytes):
            key_bytes = data_bytes[offset:offset + 4]
            value_bytes = data_bytes[offset + 4:offset + data_item_size]
            
            # 버퍼 복사 후 텐서 생성
            key_array = np.frombuffer(key_bytes, dtype=np.int32).copy()  # 읽기 전용 버퍼 복사
            key = torch.tensor(key_array.item(), dtype=torch.int32)

            value_array = np.frombuffer(value_bytes, dtype=np.float32).copy()  # 읽기 전용 버퍼 복사
            value = torch.tensor(value_array.item(), dtype=torch.float32)
            
            data[key] = value.clone().detach()
            offset += data_item_size
        self.data = data
    def get_data(self):
        return self.data
    def get_bitmask(self):
        return self.bitmask
    def is_same(self, other_message):
        other_data = other_message.get_data()
        for (key, value), (other_key, other_value) in zip(self.data.items(), other_data.items()):
            if key != other_key or value != other_value:
                return False
        return True
    def get_size(self):
        def get_dict_size(d):
            total_size = 0  
            for key, value in d.items():
                total_size += torch.tensor(key, dtype=torch.int32).element_size()
                total_size += torch.tensor(value, dtype=torch.float32).element_size()
            return total_size

        return get_dict_size(self.data) + math.ceil(self.num_clients / 8)
    
    @classmethod
    def from_packed_bytes(cls, packed_bytes, num_clients):
        # packed_bytes로부터 객체 생성 및 반환
        # unpack 메서드를 통해 초기화
        obj = cls({}, num_clients)  # 임시로 빈 데이터로 초기화
        obj.unpack(packed_bytes)
        return obj




class DZO_FLOODgossip_Client(Client):
    def __init__(self, args, model, idx, device, comm_rref):
        super().__init__(args, model, idx, device, comm_rref)
        self.current_msgs = []
        self.previous_msgs = []

        
    def local_iteration(self):
        self.total_round += 1
        self.DZO_local_iter()
    def communicate(self):
        self.DZO_flood_communicate()
    def DZO_local_iter(self):
        
        lr = self.args.lr * 1/self.args.num_clients
        framework = MeZOFramework(self.model, args=self.args, lr=lr, candidate_seeds=None)
        self.model.eval()
        self.model = self.model.to(self.device)
        loss_total_train  = 0
        
        seed_scalar = {}
        
        for _ in range(self.args.local_iter):
            try:
                batch = next(self.train_iterator)
                self.total_batch +=1
            except StopIteration:
                self.train_iterator = iter(self.train_loader)
                self.total_epoch +=1
                batch = next(self.train_iterator)
                self.total_batch +=1
            logits, loss = framework.zo_step(batch, local_seed_pool=seed_scalar)
        update_message = message(seed_scalar, self.args.num_clients)
        update_message.set_bit(self.idx)
        self.current_msgs.append(update_message)
        
        
        
        #if (not torch.isnan(loss)) and (loss != 0.0):
            #loss_total_train += loss
        
        if self.idx == 0:
            logging.info(f"client idx : {self.idx} loss : {loss} total batch {self.total_batch}, total epoch :{self.total_epoch} total_comm : {self.total_communication}")
        #logging.info(f"client {self.idx} made update : {update_message.get_data()}")
    def DZO_flood_communicate(self):
        self.current_msgs.extend(self.previous_msgs)        
        packed_items = []
        for msg in self.current_msgs:
            packed_items.append(msg.pack())
        self.comm_rref.rpc_sync().update_data(self.idx, packed_items)
        
        '''
        for msg in self.current_msgs:
            logging.info(f"client {self.idx} send update : {msg.get_data()}")
        '''
        self.current_msgs = []
        
    def upload_model(self):
        self.comm_rref.rpc_sync().update_data(self.idx,{name: param for name, param in self.model.cpu().named_parameters()})




    def gossip_result_update(self):
        packs = self.comm_rref.rpc_sync().get_data(self.idx)    

        recived_msgs = []
        for packed_bytes in packs:
            recived_msgs.append(message.from_packed_bytes(packed_bytes, self.args.num_clients))        
        '''
        for msg in recived_msgs:
            logging.info(f"client {self.idx} recieved msg : {msg.get_data()}")
        self.current_msgs = []
        '''
        
        ## merge duplicated message
        unique_recived_msgs = []
        for msg in recived_msgs:
            duple = False
            for unique_msg in unique_recived_msgs:
                if msg.is_same(unique_msg):
                    duple = True
                    unique_msg.sum_bit(msg.get_bitmask())
                    break
            if not duple:
                unique_recived_msgs.append(msg)
        
        ## ignore msg which recived previous round.
        valid_unique_recived_msgs = []
        for msg in unique_recived_msgs:
            invalid = False
            for prev_msg in self.previous_msgs:
                if msg.is_same(prev_msg):
                    invalid = True
                    break
            if not invalid:
                valid_unique_recived_msgs.append(msg)
        # set bitmask about this client, and update model
        framework = MeZOFramework(self.model, args=self.args, lr=self.args.lr * 1/self.args.num_clients, candidate_seeds=None)
        self.model.to(self.device)
        for msg in valid_unique_recived_msgs:
            assert not msg.check_bit(self.idx)
            msg.set_bit(self.idx)
            for seed, grad in msg.get_data().items():
                framework.zo_update(seed=seed.item(), grad=grad.item())
        del framework
        
        '''
        for msg in valid_unique_recived_msgs:
            logging.info(f"client {self.idx} update modle by msg : {msg.get_data()}")
        '''
        
        
        self.previous_msgs = valid_unique_recived_msgs
        
        
        return self.total_epoch, self.total_batch, self.total_communication


class DZO_FLOODgossip_Communicator(object):
    def __init__(self, args):
        self.m = generate_P(args.topology, args.num_clients)
        self.args = args
        self.data = [[] for _ in range(args.num_clients)]        
        self.model_dict = [{} for _ in range(args.num_clients)]
        self.total_comm = 0
        
        self.edge_counter = 0
        for i in range(self.args.num_clients):
            for j in range(self.args.num_clients):
                if self.m[i][j] != 0 and i != j:
                    self.edge_counter += 1
    def consensus_distance(self):
        avg_model_dict = self.make_avg_model()
        consensus_distance = 0
        for i in range(self.args.num_clients):
            for key in avg_model_dict.keys():
                consensus_distance += torch.norm(avg_model_dict[key] - self.model_dict[i][key]) ** 2
        consensus_distance /= self.args.num_clients        
        return consensus_distance ** 1/2
    
    def make_avg_model(self):
        avg_state_dict = {key: torch.zeros_like(tensor) for key, tensor in self.model_dict[0].items()}            
        for state_dict in self.model_dict:
            for key in state_dict.keys():
                avg_state_dict[key] += state_dict[key]
        
        for key in avg_state_dict.keys():
            avg_state_dict[key] /= self.args.num_clients
        return avg_state_dict
        
    def all_reduce(self):
        updated_data = [[] for _ in range(self.args.num_clients)]
        for sender in range(self.args.num_clients):
            for reciver in range(self.args.num_clients):
                for msg in self.data[sender]:
                    if self.m[sender][reciver] != 0 and not msg.check_bit(reciver):
                        assert sender != reciver
                        updated_data[reciver].append(msg)
                        self.total_comm += msg.get_size()
        self.data = updated_data
            
    def update_data(self, idx, packs):
        self.data[idx] = []
        for packed_bytes in packs:
            self.data[idx].append(message.from_packed_bytes(packed_bytes, self.args.num_clients))
    def update_model_dict(self, idx, model_dict):
        self.model_dict[idx] = model_dict
    
    def get_data(self, idx):
        pack = []
        for msg in self.data[idx]:
            pack.append(msg.pack())
        return pack
    def get_communication(self):
        return self.total_comm / self.edge_counter
    
    
if __name__ == '__main__':
    msg1 = message({1:0.1, 2:0.2, 3:0.3}, 999)
    msg1.set_bit(0)
    print(msg1.get_size())
    packedbyte = msg1.pack()
    print(len(packedbyte))
    print(packedbyte)
    new_msg = message.from_packed_bytes(packedbyte, 999)
    print(new_msg.data)
    print(new_msg.bitmask)
    print(msg1.is_same(new_msg))