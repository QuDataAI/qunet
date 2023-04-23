import os, math, copy, time, datetime
import numpy as np, matplotlib.pyplot as plt
import torch, torch.nn as nn
from   tqdm.auto import tqdm

#---------------------------------------------------------------------------

def plot_histogram( x, x_sub, pref="", digits=1, w=12, h=3, bins=50, bins_sub=100):
    """ 
    Distribution of values of x and its subset x_sub (in a narrower range) 
    """
    r = lambda x: '{x:.{digits}f}'.format(x=round(x,digits), digits=digits)
    plt.figure(figsize=(w,h), facecolor ='w')         
    plt.suptitle(f"{pref}median={r(x.median())}; mean={r(x.mean())} ± {r(x.std())}  [min,max]=[{r(x.min())}, {r(x.max())}]; cnt={len(x)} ({100*len(x_sub)/len(x):.0f}%)", fontsize=14)
    plt.subplot(1,2,1)
    plt.hist(x, bins=bins, log=True, color="lightblue", ec="black");  plt.grid(ls=":",alpha=1); plt.ylabel("log10(N)")
    plt.subplot(1,2,2)
    plt.hist(x_sub, bins=bins_sub, density=True, color="lightblue", ec="black");    plt.grid(ls=":",alpha=1); plt.ylabel("Density")
    plt.show()

#---------------------------------------------------------------------------

def plot_history(hist, view):
    """
    """    
    val, trn, labels = hist.get('val', []), hist.get('trn', []), hist.get('labels, []')
    samples, steps             = hist.get('samples', 0), hist.get('steps', 0)    

    t_unit = view.get('time_units','s') 
    t_unit_scale = dict(ms=1e-3, s=1, m=60, h=3600).get(t_unit, 1)
    c_unit = view.get('count_units', 1)
    c_unit_power = round(np.log10(c_unit), 0)
    
    plt.figure(figsize=(view.get('w', 12),  view.get('h', 5)), facecolor ='w')
    
    lr     = f"{trn['lr'][-1]:.1e}" if len(trn['lr']) else '?'
    bs_trn = f"{trn['batch_size'][-1]}" if len(trn['batch_size']) else '?'
    bs_val = f"{val['batch_size'][-1]}" if len(val['batch_size']) else '?'
    tm_trn = f"{c_unit * trn['time'][-1]/(t_unit_scale*trn['samples_epoch'][-1]):.2f}" if len(trn['time']) and trn['samples_epoch'][-1] > 0 else '?'
    tm_val = f"{c_unit * val['time'][-1]/(t_unit_scale*val['samples_epoch'][-1]):.2f}" if len(val['time']) and val['samples_epoch'][-1] > 0 else '?'

    plt.suptitle(fr"samples={samples}, steps:{steps}; lr={lr}; batch=(trn:{bs_trn}, val:{bs_val}); time=(trn:{tm_trn}, val:{tm_val}){t_unit}/$10^{c_unit_power:.0f}$; params={hist.get('parms',0)/1e3:.0f}k", fontsize = 10)

    if  not view['loss'].get('show', True):
        subplot_history(111, val, trn, view=view.get('score',{}), x_min=view.get('x_min',0), x_max=view.get('x_max',None), c_unit=c_unit, c_unit_power=c_unit_power, labels=hist.get('labels',[]), kind='score')
    elif not view['score'].get('show', True):
        subplot_history(111, val, trn, view=view.get('loss',{}),  x_min=view.get('x_min',0), x_max=view.get('x_max',None), c_unit=c_unit, c_unit_power=c_unit_power, labels=hist.get('labels',[]), kind='loss')
    else:
        subplot_history(121, val, trn, view=view.get('score',{}), x_min=view.get('x_min',0), x_max=view.get('x_max',None), c_unit=c_unit, c_unit_power=c_unit_power, labels=hist.get('labels',[]), kind='score')
        subplot_history(122, val, trn, view=view.get('loss',{}),  x_min=view.get('x_min',0), x_max=view.get('x_max',None), c_unit=c_unit, c_unit_power=c_unit_power, labels=hist.get('labels',[]), kind='loss')            
    plt.show()

    #---------------------------------------------------------------------------    

def subplot_history(sub, val, trn, view, x_min, x_max, c_unit, c_unit_power, labels, kind):                
    
    ax1 = plt.subplot(sub); ax1.grid(ls=':')                           
    if len(val['samples']) > 0 and len(trn['samples']) > 0:        
        x_max = max(val['samples'][-1], trn['samples'][-1]) if x_max is None else x_max        
        ax1.set_xlim(x_min/c_unit, x_max/c_unit)

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

    y_min, y_max = view.get('y_min'), view.get('y_max')
    ax1.set_xlabel(fr"$10^{c_unit_power:.0f}$ samples"); ax1.set_ylabel(kind);                     
    if y_min is not None  and y_max is not None:            
        ax1.set_ylim(y_min, y_max)
        if view.get('ticks', False):
            ax1.set_yticks(np.linspace(y_min, y_max, view['ticks']))  

    if len(trn['samples']):                      # trn
        ax1.plot(np.array(trn['samples'])/c_unit, trn[kind], 'darkblue', linewidth=0.5)

    if len(val['samples']):                      # val
        ax1.plot(np.array(val['samples'])/c_unit, val[kind], 'g', linewidth=1.5)

    ax1.legend([kind+'_trn',  kind+'_val'], loc='upper left', frameon = False)
    ax1.tick_params(axis='y', colors='darkgreen')

    if view.get('labels', False):
        y_min,y_max = ax1.get_ylim()
        for lb in labels:                 
            ax1.vlines(lb[1]/c_unit, y_min,y_max, linestyles=':', color='gray')
            ax1.text  (lb[1]/c_unit, y_min+(y_max-y_min)*0.01,    lb[0])


    if view.get('trn_checks', False) and len(trn['best_'+kind]):        
        for c in trn['best_'+kind]:
            if True:
                ax1.scatter(c[0]/c_unit, c[2], s=3, c='darkblue', edgecolors='black', linewidths=0.5)                                

    if view.get('val_checks', False) and  len(val['best_'+kind]):        
        for c in val['best_'+kind]:
            if True:
                ax1.scatter(c[0]/c_unit, c[2], s=7, c='g', edgecolors='black', linewidths=0.5)                                

    if view.get('lr', False):
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
            ax2.plot(np.array(trn['samples'])/c_unit, trn['lr'], ":", color='darkred')                 
        ax2.legend(['lr'], loc='upper right', frameon = False)
        ax2.tick_params(axis='y', colors='darkred')
        if sub == 121:
            ax2.set_yticklabels([])   
            ax2.minorticks_off() # for log scale         

#---------------------------------------------------------------------------

