import os, gc, math, copy, time, datetime
import numpy as np, matplotlib.pyplot as plt
import torch, torch.nn as nn
from   tqdm.auto import tqdm

class Data:
    """
    Класс даных. Может в задаче переопределяется.
    Для работы тренера Trainer в итераторе __next__ экземпляр Data должен возвращать кортеж X,Y где 
        X - тензор или кортеж тензоров для входа модели,
        Y - тензор или кортеж тензоров для целевых значений модели.
    Модель - это класс (наследник nn.Module) с методом forward и metrics.
    Первый даёт тензор или список (кортеж) тензоров на выходе модели
    Второй вычсляет ошибку (скаляр) и метрики качества (тензор для каждого примера)

    Кортеж батча данных используется в тренере следуюшим образом:
        for X,Y in data:  # при обучении
            y = model(X)
            loss, score = model.metrics(y, Y)

        y = model(X) # в продакшене

    необходимых для модели данных (батча), отправленных на нужное устройство device.

    Таким образом, dataset это список или кортеж из двух элементов (ввод и цель).
    Каждый элемент может быть тензором или списком (кортежем) тензоров.    
        Data(dataset = (X,Y) )               # 2 тензора (ввод и цель)
        Data(dataset = ( (X1,X2), Y) )       # ввод список тензоров, цель -тензор
        Data(dataset = ( (X1,X2), (Y1,Y2)) ) # ввод список тензоров, цель -список тензоров
    Предполагается, что все тензоры в dataset имеют одинаковую длину (по первому индексу).

    С Trainer можно также  использовать стандартный DataLoader:
        from torchvision            import datasets
        from torchvision.transforms import ToTensor 
        from torch.utils.data       import DataLoader

        mnist = datasets.MNIST(root='data', train=True,  transform=ToTensor(), download=True)
        print(mnist.data.shape, mnist.targets.shape, mnist.classes)
        plt.imshow(255-mnist.data[0], cmap='gray');

        data_trn  = DataLoader(dataset=mnist, batch_size=1024, shuffle=True)
    """
    def __init__(self, dataset, shuffle=True, batch_size=64,  whole_batch=False, n_packs=1) -> None:
        assert type(dataset) is list or type(dataset) is tuple, f"data = [X, Y, ...] <- list or tuple; got: {type(dataset)}"
        assert len(dataset) == 2,      "dataset must be (X,Y)"
        self.X = dataset[0]             
        self.Y = dataset[1]

        self.shuffle    = shuffle      # перемешивать или нет данные        
        self.batch_size = batch_size   # размер батча        
        self.start = 0                 # индекс начала текущего батча
        self.n_packs = n_packs         # разбить весть датасет на packs паков
        self.pack_id = 0               # номер текущего пака
        self.whole_batch = whole_batch # брать только батчи размера batch_size

        # число примеров todo: проверь, что одинаковое для всех в списках
        self.X_list = type(self.X) is list or type(self.X) is tuple
        self.Y_list = type(self.Y) is list or type(self.Y) is tuple

        self.N = len(self.X[0]) if self.Y_list else len(self.X)
        assert self.N == len(self.Y[0]) if self.Y_list else len(self.Y), "X, Y must have the same number of examples"        

    #---------------------------------------------------------------------------

    def mix(self):
        """ Перемешать данные. С list по памяти эффективнее, чем с tuple."""
        idx = torch.randperm( self.N )

        if self.X_list:
            for i in range(len(self.X)):
                self.X[i] = self.X[i][idx]        
        else:
            self.X = self.X[idx]        

        if self.Y_list:
            for i in range(len(self.Y)):
                self.Y[i] = self.Y[i][idx]        
        else:
            self.Y = self.Y[idx]        

    #---------------------------------------------------------------------------

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

        if self.X_list:
            X = [d[s: s+B] for d in self.X]
        else:
            X = self.X[s: s+B]

        if self.Y_list:
            Y = [d[s: s+B] for d in self.Y]
        else:
            Y = self.Y[s: s+B]


        return (X, Y)                

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
