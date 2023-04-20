import os, math, copy, time, datetime
import numpy as np, matplotlib.pyplot as plt
import torch, torch.nn as nn
from   tqdm.auto import tqdm

class Plotter:
    def hist(self, x, x_sub, pref="", digits=1, w=12, h=3, bins=50, bins_sub=100):
        """ Distribution of values of x and its subset x_sub (in a narrower range) """
        r = lambda x: '{x:.{digits}f}'.format(x=round(x,digits), digits=digits)
        plt.figure(figsize=(w,h), facecolor ='w')         
        plt.suptitle(f"{pref}median={r(x.median())}; mean={r(x.mean())} ± {r(x.std())}  [min,max]=[{r(x.min())}, {r(x.max())}]; cnt={len(x)} ({100*len(x_sub)/len(x):.0f}%)", fontsize=14)
        plt.subplot(1,2,1)
        plt.hist(x, bins=bins, log=True, color="lightblue", ec="black");  plt.grid(ls=":",alpha=1); plt.ylabel("log10(N)")
        plt.subplot(1,2,2)
        plt.hist(x_sub, bins=bins_sub, density=True, color="lightblue", ec="black");    plt.grid(ls=":",alpha=1); plt.ylabel("Density")
        plt.show()

    #---------------------------------------------------------------------------
    def plot(self, cfg, model, data, w=12, h=5):
        hist_val, hist_trn, labels = data.get('hist_val'), data.get('hist_trn'), data.get('labels')
        samples, steps             = data.get('samples'), data.get('steps')
        parms1 = sum(p.numel() for p in model.parameters() if     p.requires_grad)    
        #parms2 = sum(p.numel() for p in self.model.parameters() if not p.requires_grad)    
        val, trn, unit =  hist_val, hist_trn, 10**cfg['samples_unit_power']
        
        plt.figure(figsize=(w,h), facecolor ='w')
        
        lr     = f"{trn['lr'][-1]:.1e}" if len(trn['lr']) else '?'
        bs_trn = f"{trn['batch_size'][-1]}" if len(trn['batch_size']) else '?'
        bs_val = f"{val['batch_size'][-1]}" if len(val['batch_size']) else '?'
        tm_trn = f"{unit * trn['time'][-1]/trn['samples_epoch'][-1]:.2f}" if len(trn['time']) and trn['samples_epoch'][-1] > 0 else '?'
        tm_val = f"{unit * val['time'][-1]/val['samples_epoch'][-1]:.2f}" if len(val['time']) and val['samples_epoch'][-1] > 0 else '?'

        plt.suptitle(fr"samples={samples}, steps:{steps}; lr={lr}; batch=(trn:{bs_trn}, val:{bs_val}); time=(trn:{tm_trn}, val:{tm_val})s/$10^{cfg['samples_unit_power']}$; params={parms1/1e3:.0f}k", fontsize = 10)

        if  not cfg['plot_loss'].get('show', True):
            self.subplot(111, hist_val, hist_trn, cfg=cfg['plot_score'], unit=unit, x_min=cfg['x_min'], x_max=cfg['x_max'], power=cfg['samples_unit_power'], kind='score')
        elif not cfg['plot_score'].get('show', True):
             self.subplot(111, hist_val, hist_trn, cfg=cfg['plot_loss'], unit=unit, x_min=cfg['x_min'], x_max=cfg['x_max'], power=cfg['samples_unit_power'], kind='loss')
        else:
            self.subplot(121, hist_val, hist_trn, cfg=cfg['plot_score'], unit=unit, x_min=cfg['x_min'], x_max=cfg['x_max'], power=cfg['samples_unit_power'], kind='score')
            self.subplot(122, hist_val, hist_trn, cfg=cfg['plot_loss'],  unit=unit, x_min=cfg['x_min'], x_max=cfg['x_max'], power=cfg['samples_unit_power'], kind='loss')            
        plt.show()

    #---------------------------------------------------------------------------    

    def subplot(self, sub, val, trn, cfg, unit, x_min, x_max, power, kind):                
        ax1 = plt.subplot(sub); ax1.grid(ls=':')                                            
        if len(val['samples']) > 0 and len(trn['samples']) > 0:        
            x_max = max(val['samples'][-1], trn['samples'][-1]) if x_max is None else x_max
            #ax1.set_xlim(0.99*x_min/unit, x_max*1.1/unit)
            ax1.set_xlim(x_min/unit, x_max/unit)

        if kind == 'loss':
            best_loss  = f"{val['best_loss'] [-1][-1]:.4f}" if len(val['best_loss'])  else "?"
            loss_val   = f"{val['loss'] [-1]:.4f}"      if len(val['loss'])  else "?"
            loss_trn   = f"{trn['loss'] [-1]:.4f}"      if len(trn['loss'])  else "?"
            plt.title(f"loss = (min: {best_loss}, val: {loss_val}, trn: {loss_trn})")
        if kind == 'score':
            best_score = f"{val['best_score'][-1][-1]:.4f}" if len(val['best_score']) else "?"
            score_val  = f"{val['score'][-1]:.4f}"      if len(val['score']) else "?"
            score_trn  = f"{trn['score'][-1]:.4f}"      if len(trn['score']) else "?"
            plt.title(f"score = (bst: {best_score}, val: {score_val},  trn: {score_trn})")

        ax1.set_xlabel(fr"$10^{power}$ samples"); ax1.set_ylabel(kind);                     
        if cfg.get('y_min') is not None  and cfg.get('y_max') is not None:            
            ax1.set_ylim(cfg['y_min'], cfg['y_max'])
            if cfg['ticks'] is not None:
                ax1.set_yticks(np.linspace(cfg['y_min'], cfg['y_max'], cfg['ticks']))  

        if len(trn['samples']):                      # trn
            ax1.plot(np.array(trn['samples'])/unit, trn[kind], 'darkblue', linewidth=0.5)

        if len(val['samples']):                      # val
            ax1.plot(np.array(val['samples'])/unit, val[kind], 'g', linewidth=1.5)

        ax1.legend([kind+'_trn',  kind+'_val'], loc='upper left', frameon = False)
        ax1.tick_params(axis='y', colors='darkgreen')

        if len(trn['best_'+kind]):        
            for c in trn['best_'+kind]:
                if True:
                    ax1.scatter(c[0]/unit, c[2], s=3, c='darkblue', edgecolors='black', linewidths=0.5)                                

        if len(val['best_'+kind]):        
            for c in val['best_'+kind]:
                if True:
                    ax1.scatter(c[0]/unit, c[2], s=7, c='g', edgecolors='black', linewidths=0.5)                                

        ax2 = ax1.twinx();                                     # lr
        ax2.set_yscale('log')        
        if len(trn['samples']):        
            lr1, lr2 = np.min(trn['lr']), np.max(trn['lr'])
            if lr1 > 0 and lr2 > 0:
                lr1, lr2 = math.floor(np.log10(lr1)), math.ceil(np.log10(lr2))
                if lr1 == lr2:
                    lr1 -= 1; lr2 +=1
                lr1, lr2 = 10**lr1, 10**lr2
                ax2.set_ylim(lr1, lr2)
            ax2.plot(np.array(trn['samples'])/unit, trn['lr'], ":", color='darkred')                 
        ax2.legend(['lr'], loc='upper right', frameon = False)
        ax2.tick_params(axis='y', colors='darkred')
        if sub == 121:
            ax2.set_yticklabels([])   
            ax2.minorticks_off() # for log scale         

    #---------------------------------------------------------------------------

    def save(self, fname, model = None, optim=None, info=""):
        model = model or self.model
        cfg = model.cfg
        state = {
            'info':            info, 
            'date':            datetime.datetime.now(),   # дата и время
            'config':          cfg,
            'model' :          model.state_dict(),        # параметры модели         
            'optimizer':       optim.state_dict() if optim is not None else None,
            'hist_trn':        self.hist_trn,
            'hist_val':        self.hist_val,
            'labels':          self.labels,             
            'best_loss_val':   self.best_loss_val,
            'best_loss_trn':   self.best_loss_trn,                          
            'best_score_val':  self.best_score_val,
            'best_score_trn':  self.best_score_trn,
            'steps':           self.steps,
            'samples':         self.samples
        }    
        torch.save(state, fname)
        
