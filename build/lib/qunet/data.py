import os, gc, math, copy, time, datetime
import numpy as np, matplotlib.pyplot as plt
import torch, torch.nn as nn
from   tqdm.auto import tqdm

class Data:
    """
    Класс даных. Обычно в задаче переопределяется.
    Для работы тренера Trainer в итераторе __next__ должен возвращать кортеж
    необходимых для модели данных (батча), отправленных на нужное устройство device.
    Ниже пример для задачи Y=f(X). 
    
    В простых случаях, можно использовать стандартный DataLoader:
    from torchvision            import datasets
    from torchvision.transforms import ToTensor 
    from torch.utils.data       import DataLoader

    mnist = datasets.MNIST(root='data', train=True,  transform=ToTensor(), download=True)
    print(mnist.data.shape, mnist.targets.shape, mnist.classes)
    plt.imshow(255-mnist.data[0], cmap='gray');

    data_trn  = DataLoader(dataset=mnist, batch_size=1024, shuffle=True)
    """
    def __init__(self, dataset, shuffle=True, batch_size=64, device='cpu', whole_batch=False, n_packs=1) -> None:
        assert type(dataset) is list or type(dataset) is tuple, f"data = [X, Y, ...] <- list or tuple; got: {type(dataset)}"
        assert len(dataset),           "empty data"
        self.data = [d for d in dataset] # мешать будем поочереди (tuple не годится!?)
        self.shuffle    = shuffle      # перемешивать или нет данные
        self.device     = device       # на какое устройство отправлять
        self.batch_size = batch_size   # размер батча
        self.N     = len(dataset[0])   # число примеров todo: проверь, что одинаковое для всех
        self.start = 0                 # индекс начала текущего батча
        self.n_packs = n_packs         # разбить весть датасет на packs паков
        self.pack_id = 0               # номер текущего пака
        self.whole_batch = whole_batch # брать только батчи размера batch_size
        print(f"data:{[d.shape for d in self.data]}")

    def mix(self):
        """ Перемешать данные. С list по памяти эффективнее, чем с tuple."""
        idx = torch.randperm( self.N )
        for i in range(len(self.data)):
            self.data[i] = self.data[i][idx]        

    def __next__(self):        
        if (self.start >= self.N ) \
           or                      \
           (self.whole_batch and self.start + self.batch_size > self.N ):
                self.start = self.pack_id = 0                
                if self.shuffle:
                    self.mix()
                raise StopIteration

        n = self.N // self.n_packs
        if self.start > self.pack_id * n + n:
            self.pack_id += 1
            raise StopIteration

        s, B = self.start, self.batch_size
        self.start += self.batch_size
        return tuple([ d[s: s+B].to(self.device) for d in self.data] )                

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
