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
        self.done_samples = 0
        self.done_epochs  = 0
        self.lr     = 0

    #---------------------------------------------------------------------------

    def set(self, lr1=None, lr2=None, epochs=None, samples=None,  enable=True):                
        assert samples is not None or epochs is not None, f"You should define samples:{samples} or  epochs:{epochs}"
        assert samples is None or samples > 1,  f"Wrong samples={samples} limits in Scheduler"        
        assert epochs  is None or epochs  > 1,  f"Wrong epochs ={epochs}  limits in Scheduler"        

        self.lr1     = lr1
        self.lr2     = lr2    
        self.epochs  = epochs
        self.samples = samples                
        self.enable  = enable

        self.done_epochs  = 0
        self.done_samples = 0                

        if enable:
            self.set_lr(self.lr1)

    #---------------------------------------------------------------------------

    def set_lr(self, lr):
        if self.optim is not None and lr is not None:
            for g in self.optim.param_groups:
                g['lr'] = lr

    #---------------------------------------------------------------------------

    def get_lr(self):        
        if self.optim is None:
            return self.lr
        else:
            return self.optim.param_groups[0]['lr']    # а если не одна группа?

    #---------------------------------------------------------------------------

    def done(self, new_epochs, new_samples):
        if not self.enable:
            return True
        
        self.done_epochs  += new_epochs
        self.done_samples += new_samples

        if self.epochs is not None and self.done_epochs >= self.epochs:               
            self.enable = False
            return True

        if self.samples is not None and self.done_samples >= self.samples:               
            self.enable = False
            return True

        return False

    #---------------------------------------------------------------------------

    def step(self, epochs, samples):
        if self.enable:
            if self.lr1 is None:
                self.lr1 = self.get_lr()
            if self.lr2 is None:
                self.lr2 = self.get_lr()

            assert self.lr1 >= 0 and self.lr2 >= 0, f"Wrong lr1={self.lr1} or lr2={self.lr2} limits in Scheduler"                

            self.lr = self.new_lr(epochs, samples)
            self.set_lr(self.lr)
        return self.lr

    #---------------------------------------------------------------------------

    def plot(self, samples=1000, epochs = 100,  log=True, w=8, h=5, title=""):    
        """ 
        Draw a learning curve for `samples` passed over `epochs` epochs 
        Args:
            * samples - number of examples the optimizer must pass
            * epochs  - number of `epochs` (optimizer steps)
            * log - logarithmic y-axis scale
            * w  - chart width
            * h  - chart height
            * title - chart title
        """
        done_epochs, done_samples = self.done_epochs, self.done_samples
        self.done_epochs, self.done_samples = 0, 0
        s = int(samples/epochs)
        lr =   np.array([(self.lr1, 0)] + [ (self.step(1, s), i+1) for i in range(epochs)] )
        plt.figure(figsize=(w,h), facecolor ='w')        
        plt.title(title)
        plt.plot(lr[:,1], lr[:,0], "-o", markersize=3); 
        if log: plt.yscale('log')        
        plt.xlabel("epoch"); plt.ylabel("lr"); plt.grid(ls=":");         
        plt.show()
        self.done_epochs, self.done_samples = done_epochs, done_samples

    #---------------------------------------------------------------------------

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
            sch.done_epochs = 0
            sch.done_samples = 0
            sch.enable = True
        s = int(samples/epochs)

        sch = schedulers[0]
        hist, sch_id =  [ (sch.lr1, 0) ], 0
        for epoch in range(epochs):            
            if sch is not None:
                lr = sch.step(1, s)  
            hist += [ (lr, epoch+1) ]
            if sch is not None and not sch.enable:
                sch_id += 1                
                sch = schedulers[sch_id] if sch_id < len(schedulers) else None

        hist = np.array(hist)
        plt.figure(figsize=(w,h), facecolor ='w')        
        plt.title(title)
        plt.plot(hist[:,1], hist[:,0], "-o", markersize=3); 
        if log: plt.yscale('log')        
        plt.xlabel("epoch"); plt.ylabel("lr"); plt.grid(ls=":");         
        plt.show()
        
        for i in range(len(schedulers)):             # restore
            schedulers[i] = copy.deepcopy(save[i])            

#===============================================================================

class Scheduler_Const(Scheduler):
    def __init__(self, lr1:float=None, epochs=None, samples=None, enable:bool=True) -> None:
        """ 
        The scheduler sets the learning rate to lr1 and waits for `samples` samples

        Args:
            * lr1 - learning rate; if None, then the current rate is taken from the optimizer                        
            * epochs  - number of training epochs  to wait with  learning rate lr1 (if samples==None)
            * samples - number of training samples to wait with  learning rate lr1 (if epochs==None)
        """
        super().__init__()
        self.set(lr1=lr1, epochs=epochs,  samples=samples, enable=enable)

    #---------------------------------------------------------------------------

    def new_lr(self, new_epochs, new_samples):            
        self.done(new_epochs, new_samples)
        return self.lr1

#===============================================================================

class Scheduler_Line(Scheduler):
    def __init__(self, lr1:float=None, lr2:float=None, epochs=None, samples=None, enable:bool=True) -> None:
        """ 
        Linear interpolation between lr1 and lr2 per sample samples
        The logarithmic scale will be curved.            
        
        Args:
            * lr1 - initial learning rate; if None, then the current rate is taken from the optimizer            
            * lr2 - final learning rate after `samples` samples, starting from start
            * epochs  - number of training epochs  to change the learning rate from lr1 to lr2 (if samples==None)         
            * samples - number of training samples to change the learning rate from lr1 to lr2 (if epochs ==None)           
        """
        super().__init__()
        self.set(lr1=lr1, lr2=lr2, epochs=epochs, samples=samples, enable=enable)

    #---------------------------------------------------------------------------

    def new_lr(self, new_epochs, new_samples):            
        if self.done(new_epochs, new_samples):        
            return self.lr2
        else:
            count = self.epochs      if self.epochs is not None else self.samples
            done  = self.done_epochs if self.epochs is not None else self.done_samples
            if count:
                return self.lr1 + done * (self.lr2-self.lr1) / count
        return 0

#===============================================================================

class Scheduler_Exp(Scheduler):
    def __init__(self, lr1:float=None, lr2:float=None, epochs=None, samples=None, enable:bool=True) -> None:
        """ 
        Exponential interpolation between lr1 and lr2 per sample samples.
        It will be a straight line on a logarithmic scale.

        Args:
            * lr1 - initial learning rate; if None, then the current rate is taken from the optimizer            
            * lr2 - final learning rate after `samples` samples, starting from start
            * epochs  - number of training epochs  to change the learning rate from lr1 to lr2 (if samples==None)          
            * samples - number of training samples to change the learning rate from lr1 to lr2 (if epochs ==None)           
        """
        super().__init__()
        self.set(lr1=lr1, lr2=lr2, epochs=epochs, samples=samples,  enable=enable)

    #---------------------------------------------------------------------------

    def new_lr(self, new_epochs, new_samples):            
        if self.done(new_epochs, new_samples):        
            return self.lr2
        else:
            count = self.epochs if self.epochs is not None else self.samples
            done  = self.done_epochs if self.epochs is not None else self.done_samples
            beta = self.lr2/self.lr1 if  self.lr1 > 0 and self.lr2 > 0 else 1
            
            if count > 0 and beta > 0:
                return self.lr1 * math.exp(math.log(beta) * done/count)
        return 0

#===============================================================================

class Scheduler_Cos(Scheduler):
    def __init__(self, lr1:float=None, lr_hot:float=None,  lr2:float=None, samples=None, epochs=None, warmup=None, enable:bool=True) -> None:
        """
        Cosine curve with linear heating.

        Args:
            * lr1 - initial learning rate; if None, then the current rate is taken from the optimizer
            * lr_hot - learning rate after warming up after `warmup` samples, starting from start
            * lr2 - final learning rate after `samples` samples, starting from start
            * epochs  - number of training epochs to change the learning rate from lr1 to lr_hot (if samples==None)
            * samples - number of training samples to change the learning rate from lr1 to lr2   (if epochs ==None)
            * warmup - number of training epochs (or samples) to warm up from lr1 to lr_hot
        """
        super().__init__()
        self.set(lr1=lr1, lr2=lr2, epochs=epochs, samples=samples,  lr_hot=lr_hot, warmup=warmup, enable=enable)

    #---------------------------------------------------------------------------        

    def set(self, lr1=None, lr2=None, samples=None,  epochs=None, lr_hot=None, warmup=None, enable=True):        
        super().set(lr1=lr1, lr2=lr2, epochs=epochs, samples=samples,  enable=enable)        
        
        assert lr_hot is not None and lr_hot > 0,   f"wrong lr_hot={lr_hot} limit in Scheduler"
        assert warmup is not None and warmup >= 0,  f"wrong  warm-up samples or epochs limits in Scheduler: warmup={warmup}"
        assert samples is None or samples > warmup, f"should be fewer warm-up samples={warmup} than the total number of samples={samples}"
        assert epochs  is None or epochs  > warmup, f"should be fewer warm-up epochs ={warmup} than the total number of epochs ={epochs}"

        self.lr_hot = lr_hot
        self.warmup = warmup        
        if enable:
            self.set_lr(self.lr1)

    #---------------------------------------------------------------------------            

    def new_lr(self, new_epochs, new_samples):            
        if self.done(new_epochs, new_samples):        
            return self.lr2
        else:
            count = self.epochs if self.epochs is not None else self.samples
            done  = self.done_epochs if self.epochs is not None else self.done_samples            

            if done >= self.warmup:           # косинусное снижение
                if count > self.warmup:
                    s = (done - self.warmup) / (count - self.warmup)
                    return self.lr2 + (self.lr_hot - self.lr2) * (1 + math.cos(math.pi*s)) / 2            
            else:                                    # режим разогрева            
                if  self.warmup > 0:
                    return self.lr1 + (self.lr_hot - self.lr1) * done / self.warmup
        return 0


#===============================================================================

class Scheduler_OneCicleRL(Scheduler_Cos):
    def __init__(self, 
                 epochs, lr_max = 1e-3,
                 div_factor=25.0, final_div_factor=10000.0,  pct_start=0.3,
                 enable:bool=True) -> None:
        """
        Scheduler_OneCicleRL with Cosine curve.

        Args:
        ------------
            * lr_max (float: 1e-3)
                max learning rate
            * div_factor (float = 25.):
                lr1 = lr_max / div_factor
            * final_div_factor (float = 10000.):
                lr2 = lr_max / final_div_factor        
            * pct_start (float=0.3)
                warmup = epochs * pct_start
        """
        lr1 = lr_max / div_factor
        lr2 = lr_max / final_div_factor
        warmup = epochs * pct_start
        super().__init__(lr1=lr1, lr2=lr2, epochs=epochs, samples=None,  lr_hot=lr_max, warmup=warmup, enable=enable)
        

#===============================================================================
#                                   Main
#===============================================================================

if __name__ == '__main__':
    scheduler = Scheduler_Cos(lr1=1e-4, lr_hot=1e-2, lr2=1e-3, epochs=40, warmup=10)                
    #scheduler = Scheduler_Exp(lr1=1e-2,  lr2=1e-6, samples=4000)                
    #scheduler = Scheduler_Line(lr1=1e-2,  lr2=1e-3, samples=4000)                
    #scheduler.plot(samples=4000, epochs=50, log=True, w=5, h=4, title="ExpScheduler")
    #scheduler.plot(samples=8000, epochs = 100)
    
    lst =[
        #Scheduler_Cos  (lr1=1e-4, lr_hot=1e-2, lr2=1e-3, samples=4000, warmup=1000),
        #Scheduler_Const(lr1=1e-3, samples=2000),
        #Scheduler_Exp  (lr1=1e-3, lr2=1e-5, epochs=20)
        Scheduler_OneCicleRL(epochs=100)
    ]
    scheduler.plot_list(lst, samples=20000, epochs=100, log=True, w=5, h=4, title="Cos, Const, Exp Schedulers")
