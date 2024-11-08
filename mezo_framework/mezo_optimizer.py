"""
The implementations of MeZO optimizer is
adapted from https://github.com/princeton-nlp/MeZO (MIT License)

Copyright (c) 2021 Princeton Natural Language Processing

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
import torch
import numpy as np

import torch.nn as nn
import logging
logging.basicConfig(level=logging.DEBUG)


class MeZOFramework(object):
    def __init__(self, model, args, lr, candidate_seeds):
        #print('FedKSeed')
        # determine which parameters to optimizes
        self.args = args
        self.lr = lr
        self.model = model
        self.named_parameters_to_optim = []
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.named_parameters_to_optim.append((name, param))

        self.zo_eps = self.args.zo_eps
        self.candidate_seeds = candidate_seeds
        
        
    def zo_step(self, batch, local_seed_pool=None):
        """
        Estimate gradient by MeZO. Return the loss from f(theta + z)
        """
        # Sample the random seed for sampling z
        projected_grad_list = []
        random_seed_list = []
        avg_loss = 0
        for _ in range(self.args.npert):
            if self.candidate_seeds is not None:
                self.zo_random_seed = int(np.random.choice(self.candidate_seeds, 1)[0])
            else:
                self.zo_random_seed = torch.randint(low=1, high=2147483647, size=(1,), dtype=torch.int32).item()
            self._zo_perturb_parameters(scaling_factor=1)
            logits1, loss1 = self.zo_forward(batch)

            # Second function evaluation
            self._zo_perturb_parameters(scaling_factor=-2)
            logits2, loss2 = self.zo_forward(batch)
            
            # Reset model back to its parameters at start of step
            self._zo_perturb_parameters(scaling_factor=1)
            
            if torch.isnan(loss1):
                return logits1, loss1
            if torch.isnan(loss2):
                return logits2, loss2
            if self.args.grad_clip > 0.0:
                if torch.abs(loss1 - loss2)  > self.args.grad_clip:
                    return logits1, 0.0
        
        
            self.projected_grad = ((loss1 - loss2) / (2 * self.zo_eps)).item()
            if self.args.zo_normalized :
                total_norm, d = self.zo_normalized(self.zo_random_seed) 
                total_norm = (total_norm**1/2) * self.projected_grad
                if total_norm > 1:
                    self.projected_grad = ((loss1 - loss2)*  ((np.pi * d/2)**1/2) / ((2 * self.zo_eps) * (total_norm))).item()  
            
            projected_grad_list.append(self.projected_grad / self.args.npert)
            random_seed_list.append(self.zo_random_seed)
        
        for projected_grad, random_seed in zip(projected_grad_list, random_seed_list):
            self.zo_random_seed = random_seed
            self.projected_grad = projected_grad
            self.zo_update()
            if local_seed_pool is not None:
                local_seed_pool[self.zo_random_seed] = local_seed_pool.get(self.zo_random_seed, 0) + self.projected_grad
        return logits1, loss1

    def _zo_perturb_parameters(self, scaling_factor=1):
        """
        Perturb the parameters with random vector z.
        Input:
        - scaling_factor: theta = theta + scaling_factor * z * eps
        """
        torch.manual_seed(self.zo_random_seed)

        for _, param in self.named_parameters_to_optim:
            z = torch.normal(mean=0,
                             std=1,
                             size=param.data.size(),
                             device=param.data.device,
                             dtype=param.data.dtype)
            param.data = param.data + scaling_factor * self.zo_eps * z

    def zo_forward(self, batch):
        """
        Get (no gradient) loss from the model. Dropout is turned off too.
        """
        #outputs = self.model(**batch)
        #logits = outputs.logits
        #loss = outputs.loss
        if self.args.model == "logistic_regression":
            criterion = nn.BCELoss()
        else:
            criterion = nn.CrossEntropyLoss()

        device = next(self.model.parameters()).device
        data, target = batch[0].to(device).float(), batch[1].to(device)
        outputs = self.model(data)
        loss = criterion(outputs, target)
        
        return 0, loss.detach() #logits.detach(), loss.detach()
    
    def zo_update(self, seed=None, grad=None):
        """
        Update the parameters with the estimated gradients.
        """

        # Reset the random seed for sampling zs
        if seed is None:
            torch.manual_seed(self.zo_random_seed)     
            for name, param in self.named_parameters_to_optim:
                # Resample z
                z = torch.normal(mean=0, std=1, size=param.data.size(), device=param.data.device, dtype=param.data.dtype)
                if "bias" not in name and "layer_norm" not in name and "layernorm" not in name:
                    param.data = param.data - self.lr * (self.projected_grad * z + self.args.weight_decay * param.data)
                else:
                    param.data = param.data - self.lr * (self.projected_grad * z)
        else:
            torch.manual_seed(seed)
            for name, param in self.named_parameters_to_optim:
                # Resample z
                z = torch.normal(mean=0, std=1, size=param.data.size(), device=param.data.device, dtype=param.data.dtype)
                if "bias" not in name and "layer_norm" not in name and "layernorm" not in name:
                    param.data = param.data - self.lr * (grad * z + self.args.weight_decay * param.data)
                else:
                    param.data = param.data - self.lr * (grad * z)
    def zo_normalized(self, seed):
        torch.manual_seed(seed)     
        total_norm = 0
        total_d = 0
        for name, param in self.named_parameters_to_optim:
            # Resample z
            z = torch.normal(mean=0, std=1, size=param.data.size(), device=param.data.device, dtype=param.data.dtype)
            total_norm += torch.norm(z) ** 2
            total_d += z.nelement()
        return total_norm, total_d 




"""

hyperparameter : 
args.model = 
args.zo_eps : epsilon in ZO
args.npert : n SPSA
args.grad_clip : dummy, set <0
args.zo_normalized : grad clipping, with root(d)
args.weight_decay: weight decay parameter

initialize 
        framework = MeZOFramework(self.model, args=self.args, lr=lr, candidate_seeds=None)

optimizer update : 
        logits, loss = framework.zo_step(batch)
        
"""