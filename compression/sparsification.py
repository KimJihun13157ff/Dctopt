# Quantize to [-k, k]

import pickle
import torch
import numpy as np

from compression.operater import Compression


class Sparsification(Compression):
    """
    Compress metadata and quantize parameters

    """

    def __init__(self, ratio:float = 0.01, op = "top_k" , *args, **kwargs):
        """
        Constructor

        Parameters
        ----------
        ratio : 
        
        float_precision : int, optional
            Quantization parameter
        """
        super().__init__(*args, **kwargs)
        self.ratio = ratio
        self.op = op
        
    def get_top_k(self, x, ratio):
        """it will sample the top 1-ratio of the samples."""
        original_shape = x.shape
        x_data = x.view(-1)
        x_len = x_data.nelement()
        top_k = max(1, int(x_len * (1 - ratio)))

        # get indices and the corresponding values
        if top_k == 1:
            _, selected_indices = torch.max(x_data.abs(), dim=0, keepdim=True)
        else:
            _, selected_indices = torch.topk(
                x_data.abs(), top_k, largest=True, sorted=False
            )
        mask, _ = self.get_mask(x_data, selected_indices)
        x_sparse = x_data * mask 
        #return x_data[selected_indices], selected_indices
        return x_sparse.view(original_shape), mask


    def get_mask(self, flatten_arr, indices):
        mask = torch.zeros_like(flatten_arr)
        mask[indices] = 1

        mask = mask.byte()
        return mask.float(), (~mask).float()

    def get_random_k(self, x, ratio, is_biased=True):
        """it will randomly sample the 1-ratio of the samples."""
        # get tensor size.
        original_shape = x.shape
        x_data = x.view(-1)
        x_len = x_data.nelement()
        top_k = max(1, int(x_len * (1 - ratio)))

        # random sample the k indices.
        selected_indices = np.random.choice(x_len, top_k, replace=False)
        selected_indices = torch.LongTensor(selected_indices).to(x.device)
        mask, _ = self.get_mask(x_data, selected_indices)
        x_sparse = x_data * mask 
        if is_biased:
            return x_sparse.view(original_shape), mask
        else:
            return x_len / top_k *x_sparse.view(original_shape), mask


    def compress_float(self, x):
        """
        compression function for float arrays

        Parameters
        ----------
        x : np.ndarray
            Data to compress

        Returns
        -------
        bytearray
            encoded data as bytes

        """
        if "top_k" in self.op:
            values, mask = self.get_top_k(x, self.ratio)
        elif "random_k" in self.op:
            values, mask = self.get_random_k(x, self.ratio)
        else:
            raise NotImplementedError
        return values, mask
    
    def decompress_float(self, bytes):
        """
        decompression function for compressed float arrays

        Parameters
        ----------
        bytes :bytearray
            compressed data

        Returns
        -------
        np.ndarray
            decompressed data as array

        """