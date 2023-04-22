import os, math, copy, time, datetime
import numpy as np, matplotlib.pyplot as plt
import torch, torch.nn as nn
from   tqdm.auto import tqdm

        
#===============================================================================
#                                   Shedulers
#===============================================================================

class Scheduler:
    def __init__(self) -> None:
        self.optim  = None
        self.enable = False        
        self.done   = 0
        self.lr     = 0

    def set(self, lr1=None, lr2=None, samples=1e3, enable=True):        
        self.lr1 = lr1
        self.lr2 = lr2    
        assert samples > 1,  "wrong samples limits in Scheduler"
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
            return self.lr
        else:
            return self.optim.param_groups[0]['lr']    # а если не одна группа?

    def step(self, samples):
        if self.enable:
            if self.lr1 is None:
                self.lr1 = self.get_lr()
            if self.lr2 is None:
                self.lr2 = self.get_lr()
            assert self.lr1 >= 0 and self.lr2 >= 0, "wrong lr limits in Scheduler"                

            self.lr = self.new_lr(samples)
            self.set_lr(self.lr)
        return self.lr

    def plot(self, samples=1000, epochs = 100,  log=True, w=8, h=5, title=""):    
        """ 
        Draw a learning curve for `samples` passed over `epochs` epochs 
        Args:
            * samples - number of examples the optimizer must pass
            * epochs - number of `epochs` (optimizer steps)
            * log - logarithmic y-axis scale
            * w  - chart width
            * h  - chart height
            * title - chart title
        """
        done = self.done
        self.done = 0
        s = int(samples/epochs)
        lr =   np.array([(self.lr1, 0)] + [ (self.step(s), self.done) for i in range(epochs)] )
        plt.figure(figsize=(w,h), facecolor ='w')        
        plt.title(title)
        plt.plot(lr[:,1], lr[:,0], "-o", markersize=3); 
        if log: plt.yscale('log')        
        plt.xlabel("epoch"); plt.ylabel("lr"); plt.grid(ls=":");         
        plt.show()
        self.done = done

#-------------------------------------------------------------------------------

    def plot_list(self, schedulers, samples=1000, epochs = 100,  log=True, w=8, h=5, title=""):
        """ 
        Draw a learning curve for `samples` passed over `epochs` epochs 
        Args:
            * schedulers - list of schedulers
            * samples - number of examples the optimizer must pass
            * epochs - number of `epochs` (optimizer steps)
            * log - logarithmic y-axis scale
            * w  - chart width
            * h  - chart height
            * title - chart title
        """
        assert len(schedulers) > 0
        save = [ copy.deepcopy(sch) for sch in schedulers ]
        for sch in schedulers:
            sch.done = 0
            sch.enable = True
        s = int(samples/epochs)

        sch = schedulers[0]
        hist, sch_id, tot =  [ (sch.lr1, 0) ], 0, 0
        for epoch in range(epochs):            
            lr = sch.step(s)            
            tot += s
            hist += [ (lr, tot) ]
            if not sch.enable:
                sch_id += 1
                if sch_id >= len(schedulers):
                    break
                sch = schedulers[sch_id]

        hist = np.array(hist)
        plt.figure(figsize=(w,h), facecolor ='w')        
        plt.title(title)
        plt.plot(hist[:,1], hist[:,0], "-o", markersize=3); 
        if log: plt.yscale('log')        
        plt.xlabel("epoch"); plt.ylabel("lr"); plt.grid(ls=":");         
        plt.show()
        
        for i in range(len(schedulers)):             # restore
            schedulers[i] = copy.deepcopy(save[i])            

#-------------------------------------------------------------------------------


class Scheduler_Const(Scheduler):
    def __init__(self, lr1:float=None, samples:int=1000, enable:bool=True) -> None:
        """ 
        The scheduler sets the learning rate to lr1 and waits for `samples` samples

        Args:
            * lr1 - learning rate; if None, then the current rate is taken from the optimizer                        
            * samples - number of training samples to wait with  learning rate lr1
        """
        super().__init__()
        self.set(lr1=lr1,  samples=samples, enable=enable)

    def new_lr(self, new_samples):            
        self.done += new_samples
        if self.done >= self.samples:               
            self.enable = False
        return self.lr1

#-------------------------------------------------------------------------------

class Scheduler_Line(Scheduler):
    def __init__(self, lr1:float=None, lr2:float=1e-5, samples:int=1000, enable:bool=True) -> None:
        """ 
        Linear interpolation between lr1 and lr2 per sample samples
        The logarithmic scale will be curved.            
        
        Args:
            * lr1 - initial learning rate; if None, then the current rate is taken from the optimizer            
            * lr2 - final learning rate after `samples` samples, starting from start
            * samples - number of training samples to change the learning rate from lr1 to lr2            
        """
        super().__init__()
        self.set(lr1=lr1, lr2=lr2, samples=samples, enable=enable)

    def new_lr(self, new_samples):            
        self.done += new_samples
        if self.done >= self.samples:   
            lr = self.lr2
            self.enable = False
        else:
            lr = self.lr1 + self.done * (self.lr2-self.lr1) / self.samples
        return lr

#-------------------------------------------------------------------------------

class Scheduler_Exp(Scheduler):
    def __init__(self, lr1:float=None, lr2:float=1e-5, samples:int=1000, enable:bool=True) -> None:
        """ 
        Exponential interpolation between lr1 and lr2 per sample samples.
        It will be a straight line on a logarithmic scale.

        Args:
            * lr1 - initial learning rate; if None, then the current rate is taken from the optimizer            
            * lr2 - final learning rate after `samples` samples, starting from start
            * samples - number of training samples to change the learning rate from lr1 to lr2            
        """
        super().__init__()
        self.set(lr1=lr1, lr2=lr2, samples=samples, enable=enable)

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

class Scheduler_Cos(Scheduler):
    def __init__(self, lr1:float=None, lr_hot:float=5e-3,  lr2:float=1e-5, samples:int=1000, warmup:int=100, enable:bool=True) -> None:
        """
        Cosine curve with linear heating.

        Args:
            * lr1 - initial learning rate; if None, then the current rate is taken from the optimizer
            * lr_hot - learning rate after warming up after `warmup` samples, starting from start
            * lr2 - final learning rate after `samples` samples, starting from start
            * samples - number of training samples to change the learning rate from lr1 to lr2
            * warmup - number of training samples to warm up from lr1 to lr_hot
        """
        super().__init__()
        self.set(lr1=lr1, lr2=lr2, samples=samples, lr_hot=lr_hot, warmup=warmup, enable=enable)

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
    #scheduler = Scheduler_Cos(lr1=1e-4, lr_hot=1e-2, lr2=1e-3, samples=4000, warmup=1000)                
    scheduler = Scheduler_Exp(lr1=1e-2,  lr2=1e-6, samples=4000)                
    #scheduler.plot(samples=4000, epochs=50, log=True, w=5, h=4, title="ExpScheduler")
    
    lst =[
        Scheduler_Cos  (lr1=1e-4, lr_hot=1e-2, lr2=1e-3, samples=4000, warmup=1000),
        Scheduler_Const(lr1=1e-3, samples=2000),
        Scheduler_Exp  (lr1=1e-3, lr2=1e-5, samples=4000)
    ]
    scheduler.plot_list(lst, samples=10000, epochs=50, log=True, w=5, h=4, title="Cos, Const, Exp Schedulers")
