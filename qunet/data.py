import os, gc, math, copy, time, datetime
import numpy as np, matplotlib.pyplot as plt
import torch, torch.nn as nn
from   tqdm.auto import tqdm

class Data:
    """
    Класс даных. Может в задаче переопределяется.
    """
    def __init__(self, dataset, shuffle=True, batch_size=64,  whole_batch=False, n_packs=1) -> None:
        assert torch.is_tensor(dataset) or type(dataset) is list or type(dataset) is tuple, f"data = tensor or [X, Y, ...] <- list or tuple; got: {type(dataset)}"
        
        self.data = dataset        
        self.shuffle    = shuffle      # перемешивать или нет данные        
        self.batch_size = batch_size   # размер батча        
        self.start = 0                 # индекс начала текущего батча
        self.n_packs = n_packs         # разбить весть датасет на packs паков
        self.pack_id = 0               # номер текущего пака
        self.whole_batch = whole_batch # брать только батчи размера batch_size

        self.N = self.count(dataset)    

    #---------------------------------------------------------------------------

    def count(self, data):
        """ Считаем количество данных """
        if torch.is_tensor(data):
            return len(data)
        if type(data) is list or type(data) is tuple and len(data):
            return self.count(data[0])
        assert False, "Wrong dataset structure: must be tensors, or not empty lists or tupes"

    #---------------------------------------------------------------------------

    def mix(self, data, idx=None):
        """ Перемешать данные. С list по памяти эффективнее, чем с tuple."""
        if idx is None:
            idx = torch.randperm( self.N )
            
        if torch.is_tensor(data):
            return data[idx]
    
        if type(data) is list or type(data) is tuple and len(data):
            if type(data) is tuple:
                data = list(data)
            for i in range(len(data)):
                data[i] = self.mix(data[i], idx)
            return data
        
        assert False, "Wrong dataset structure: must be tensors, or not empty lists or tupes"

    #---------------------------------------------------------------------------

    def get_batch(self, data, s, B):
        if torch.is_tensor(data):
            return data[s: s+B]
        
        if type(data) is list:
            return [ self.get_batch(d, s, B) for d in data ]

        if type(data) is tuple:
            return tuple([ self.get_batch(d, s, B) for d in data ])

        assert False, "Wrong dataset structure: must be tensors, or not empty lists or tupes"
    #---------------------------------------------------------------------------

    def __next__(self):        
        if (self.start >= self.N ) \
           or                      \
           (self.whole_batch and self.start + self.batch_size > self.N ):
                self.start = self.pack_id = 0                
                if self.shuffle:
                    self.data = self.mix(self.data)
                raise StopIteration

        n = self.N // self.n_packs
        if self.start > self.pack_id * n + n:
            self.pack_id += 1
            raise StopIteration
        
        batch = self.get_batch(self.data, self.start, self.batch_size)
        self.start += self.batch_size
        return batch

    def __iter__(self):
        return self

    def __len__(self):
        nb = self.N  // self.batch_size
        if not self.whole_batch and self.N  % self.batch_size:
            nb += 1
        return nb


#===============================================================================
#                                   Main
#===============================================================================
if __name__ == '__main__':
    X = torch.rand((1000, 2))
    Y = (X[:,0] * X[:,1]).view(-1,1)     # (B,1)  !!!
    n_trn = int(len(X)*0.80)
    data_trn = Data((X[:n_trn], Y[:n_trn]), batch_size=40, whole_batch=True)
    data_val = Data((X[n_trn:], Y[n_trn:]), batch_size=40, whole_batch=False, n_packs=4)
