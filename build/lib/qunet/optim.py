import os, math, copy, time, datetime
import numpy as np, matplotlib.pyplot as plt
import torch, torch.nn as nn
from   tqdm.auto import tqdm

        
#===============================================================================
#                                   Shedulers
#===============================================================================

class Scheduler:
    def __init__(self, optim=None) -> None:
        self.optim = optim
        self.enable = False
        self.lr = 0

    def set(self, lr1=1e-3, lr2=1e-5, samples=1e3, enable=True):        
        self.lr1 = lr1 or self.get_lr()
        self.lr2 = lr2 or self.get_lr()
        assert self.lr1 > 0 and self.lr2 > 0, "wrong lr limits in Scheduler"
        assert samples > 1,                   "wrong samples limits in Scheduler"
        self.samples = samples        
        self.enable  = enable
        self.done    = 0
        if enable:
            self.set_lr(self.lr1)

    def set_lr(self, lr):
        if self.optim is not None:
            for g in self.optim.param_groups:
                g['lr'] = lr
        
    def get_lr(self):        
        if self.optim is None:
            return 0
        else:
            return self.optim.param_groups[0]['lr']    # а если не одна группа?

    def step(self, samples):
        if self.enable:
            self.lr = self.new_lr(samples)
            self.set_lr(self.lr)

    def plot(self, samples=1000, epochs = 100, w=8, h=5, log=True):    
        """ Нарисовать кривую обучения для пройденных "samples" за "epochs" эпох """
        done = self.done
        self.done = 0
        s = int(samples/epochs)
        lr =   np.array([(self.lr1, 0)] + [ (self.new_lr(s), self.done) for i in range(epochs)] )
        plt.figure(figsize=(w,h), facecolor ='w')        
        plt.plot(lr[:,1], lr[:,0], "-o", markersize=3); 
        if log: plt.yscale('log')        
        plt.xlabel("epoch"); plt.ylabel("lr"); plt.grid(ls=":");         
        plt.show()
        self.done = done

#-------------------------------------------------------------------------------

class LineScheduler(Scheduler):
    """ 
    Линейная интерполяция между lr1 и lr2 за samples примеров
    В логарифической шкале будет изогнута.
    """
    def new_lr(self, new_samples):            
        self.done += new_samples
        if self.done >= self.samples:   
            lr = self.lr2
            self.enable = False
        else:
            lr = self.lr1 + self.done * (self.lr2-self.lr1) / self.samples
        return lr

#-------------------------------------------------------------------------------

class ExpScheduler(Scheduler):
    """ 
    Экспоненциальная интерполяция между lr1 и lr2 за samples примеров.
    В логарифической шкале будет прямая.
    """
    def new_lr(self, new_samples):            
        self.done += new_samples
        if self.done >= self.samples:   
            lr = self.lr2
            self.enable = False
        else:
            beta = self.lr2/self.lr1 if  self.lr1 > 0 and self.lr2 > 0 else 1
            lr = self.lr1 * math.exp(math.log(beta) * self.done/self.samples)
        return lr

#-------------------------------------------------------------------------------

class CosScheduler(Scheduler):
    """
    Косинусная кривая с разогревом.
    src: https://d2l.ai/chapter_optimization/lr-scheduler.html 
    """

    def set(self, lr1=1e-5, lr2=1e-5, samples=1e3, lr_hot=5e-3, warmup=1e1, enable=True):        
        super().set(lr1=lr1, lr2=lr2, samples=samples, enable=enable)
        self.lr_hot = lr_hot or self.get_lr()        
        assert self.lr_hot > 0,  "wrong lr limits in Scheduler"
        assert warmup >= 0,      "wrong samples limits in Scheduler"
        assert samples > warmup, "should be fewer warm-up samples (warm) than the total number of samples"
        self.warmup  = warmup
        self.done    = 0
        if enable:
            self.set_lr(self.lr1)

    def new_lr(self, new_samples):            
        self.done += new_samples
        if self.done >= self.samples:   
            lr = self.lr2
            self.enable = False
        elif self.done >= self.warmup:           # косинусное снижение
            s = (self.done - self.warmup) / (self.samples - self.warmup)
            lr = self.lr2 + (self.lr_hot - self.lr2) * (1 + math.cos(math.pi*s)) / 2            
        else:                                    # режим разогрева            
            lr =  self.lr1 + (self.lr_hot - self.lr1) * self.done / self.warmup
        return lr

#===============================================================================
#                                   Main
#===============================================================================

if __name__ == '__main__':

    scheduler = CosScheduler()        
    scheduler.set(lr1=1e-6, lr2=1e-5, samples=40000, lr_hot=1e-3, warmup=1000)
    scheduler.plot(samples=40000, log=False)
   