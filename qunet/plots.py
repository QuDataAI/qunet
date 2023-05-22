import os, math, copy, time, datetime
import numpy as np, matplotlib.pyplot as plt
import torch, torch.nn as nn
from   tqdm.auto import tqdm

#---------------------------------------------------------------------------

def plot_histogram( x, x_sub=None, pref="", digits=1, w=12, h=3, bins=50, bins_sub=100, xlim=None, fname=""):
    """ 
    Distribution of values of x and its subset x_sub (in a narrower range) 
    """
    r = lambda x: '{x:.{digits}f}'.format(x=round(x,digits), digits=digits)
    if x_sub is not None:
        _, (ax1, ax2) = plt.subplots(1,2, figsize=(w, h), facecolor ='w')    
        plt.suptitle(f"{pref}median={r(np.median(x))}; mean={r(x.mean())} ± {r(x.std())}  [min,max]=[{r(x.min())}, {r(x.max())}]; cnt={len(x)} ({100*len(x_sub)/len(x):.0f}%)", fontsize=14)    
        ax1.hist(x, bins=bins, log=True, color="lightblue", ec="black");  plt.grid(ls=":",alpha=1); plt.ylabel("log10(N)")    
        ax2.hist(x_sub, bins=bins_sub, density=True, color="lightblue", ec="black");    plt.grid(ls=":",alpha=1); plt.ylabel("Density")
    else:
        _, ax = plt.subplots(1,1, figsize=(w, h), facecolor ='w')    
        plt.title(f"{pref}median={r(np.median(x))}; mean={r(x.mean())} ± {r(x.std())} [{r(x.min())}, {r(x.max())}]; cnt={len(x)}", fontsize=14)    
        ax.hist(x, bins=bins, density=True, color="lightblue", ec="black");    plt.grid(ls=":",alpha=1); plt.ylabel("Density")
        if xlim is not None:
            ax.set_xlim(xlim)
    
    if fname:
        plt.savefig(fname, bbox_inches='tight')
    else:
        plt.show()


#---------------------------------------------------------------------------

def plot_history(hist, view, fname=""):
    """
    """    
    samples, steps             = hist.samples, hist.steps
    if samples == 0:
        print("plot_history warning: empty history")
        return
    val, trn, labels = hist.val, hist.trn, hist.labels
    
    t_unit = view.units.time  if  view.units.time in ['ms','s','m','h']  else 's'
    t_unit_scale = dict(ms=1e-3, s=1, m=60, h=3600)[t_unit]
    c_unit = view.units.count if view.units.count > 0  else 1
    c_unit_power = round(np.log10(c_unit), 0)
            
    plt.figure(figsize=(view.w,  view.h), facecolor ='w')
    
    lr     = f"{trn.lr[-1]:.1e}" if len(trn.lr) else '?'
    bs_trn = f"{trn.batch_size[-1]}" if len(trn.batch_size) else '?'
    bs_val = f"{val.batch_size[-1]}" if len(val.batch_size) else '?'
    tm_trn = f"{c_unit * trn.times[-1]/(t_unit_scale*trn.samples_epoch[-1]):.2f}" if len(trn.times) and trn.samples_epoch[-1] > 0 else '?'
    tm_val = f"{c_unit * val.times[-1]/(t_unit_scale*val.samples_epoch[-1]):.2f}" if len(val.times) and val.samples_epoch[-1] > 0 else '?'

    plt.suptitle(fr"samples={samples}, steps:{steps}; lr={lr}; batch=(trn:{bs_trn}, val:{bs_val}); time=(trn:{tm_trn}, val:{tm_val}){t_unit}/$10^{c_unit_power:.0f}$; params={hist.params/1e3:.0f}k", fontsize = 10)

    if  not view.loss.show:
        subplot_history(111, val, trn, view=view.score, view_tot=view, c_unit=c_unit, c_unit_power=c_unit_power, unit=view.units.unit, labels=labels, kind='score')
    elif not view.score.show:
        subplot_history(111, val, trn, view=view.loss,  view_tot=view, c_unit=c_unit, c_unit_power=c_unit_power, unit=view.units.unit, labels=labels, kind='loss')
    else:
        subplot_history(121, val, trn, view=view.score, view_tot=view, c_unit=c_unit, c_unit_power=c_unit_power, unit=view.units.unit, labels=labels, kind='score')
        subplot_history(122, val, trn, view=view.loss,  view_tot=view, c_unit=c_unit, c_unit_power=c_unit_power, unit=view.units.unit, labels=labels, kind='loss')            
    if fname:
        plt.savefig(fname, bbox_inches='tight')
    else:
        plt.show()

    #---------------------------------------------------------------------------    

def subplot_history(sub, val, trn, view, view_tot, c_unit, c_unit_power, unit, labels, kind):                
    
    ax1 = plt.subplot(sub); ax1.grid(ls=':')                           
    if len(val.samples) > 0 and len(trn.samples) > 0:        
        if unit == 'samples':
            x_max = max(val.samples[-1], trn.samples[-1]) if view_tot.x_max is None else view_tot.x_max        
            ax1.set_xlim(view_tot.x_min/c_unit, x_max/c_unit)
        else:
            x_max = max(val.epochs[-1], trn.epochs[-1]) if view_tot.x_max is None else view_tot.x_max        
            ax1.set_xlim(view_tot.x_min, x_max)

    if kind == 'loss':
        
        best_loss  = f"{val.best.loss:.4f}"    if val.best.loss is not None  else "?"
        loss_val   = f"{val.losses[-1]:.4f}"   if len(val.losses)  else "?"
        loss_trn   = f"{trn.losses[-1]:.4f}"   if len(trn.losses)  else "?"
        plt.title(f"loss = (min: {best_loss} [{val.best.loss_epochs}], val: {loss_val}, trn: {loss_trn})", fontsize=12, pad=-2)

    if kind == 'score':        
        best_score = f"{val.best.score:.4f}"   if val.best.score is not None else "?"
        score_val  = f"{val.scores[-1]:.4f}"   if len(val.scores) else "?"
        score_trn  = f"{trn.scores[-1]:.4f}"   if len(trn.scores) else "?"
        plt.title(f"score = (bst: {best_score} [{val.best.score_epochs}], val: {score_val},  trn: {score_trn})", fontsize=12, pad=-2)

    y_min, y_max = view.y_min, view.y_max
    ax1.set_xlabel(fr"$10^{c_unit_power:.0f}$ samples" if unit=='samples' else 'epochs'); ax1.set_ylabel(kind);                     
    if y_min is not None  and y_max is not None:            
        ax1.set_ylim(y_min, y_max)
        if view.ticks:
            ax1.set_yticks(np.linspace(y_min, y_max, view.ticks))  

    text = view.cfg.get_jaml(exclude=view.exclude)
    if text:                
        ax1.text(0.02, 0.02, text[:-1],  ha='left', va='bottom', transform = ax1.transAxes, fontsize=view.fontsize, family="monospace")

    if len(trn.samples):                      # trn
        x = np.array(trn.samples)/c_unit if unit=='sample' else np.array(trn.epochs)
        y = trn.losses if kind=='loss' else trn.scores    
        plot_smooth_line(ax1, x,y, 'darkblue', label=kind+'_trn', **view_tot.smooth.get_dict())            

    if len(val.samples):                      # val
        x = np.array(val.samples)/c_unit if unit=='sample' else np.array(val.epochs)
        y = val.losses if kind=='loss' else val.scores        
        plot_smooth_line(ax1, x,y, 'g', label=kind+'_val', **view_tot.smooth.get_dict())


    ax1.legend(loc='upper left', frameon = False)
    ax1.tick_params(axis='y', colors='darkgreen')

    if view.labels:
        y_min,y_max = ax1.get_ylim()
        for lb in labels:  
            x = lb[2]/c_unit if unit=='sample' else lb[1]               
            ax1.vlines(x, y_min,y_max, linestyles=':', color='gray', linewidths=1.5 if len(lb[0]) else 1)
            ax1.text  (x, y_min+(y_max-y_min)*0.01,    lb[0])

    checks = trn.best.losses if kind=='loss' else trn.best.scores
    if view.trn_checks:        
        if view.last_checks > 0:
            checks = checks[-view.last_checks :]
        for c in checks:            
            x = c[2]/c_unit if unit=='sample' else c[1]               
            ax1.scatter(x, c[0],  s=3, c='darkblue', edgecolors='black', linewidths=0.5)                                

    checks = val.best.losses if kind=='loss' else val.best.scores
    if view.val_checks:        
        if view.last_checks > 0:
            checks = checks[-view.last_checks :]
        for c in checks:            
            x = c[2]/c_unit if unit=='sample' else c[1]               
            ax1.scatter(x, c[0], s=7, c='g', edgecolors='black', linewidths=0.5)                                

    if view.lr and len(trn.samples):
        ax2 = ax1.twinx();                                     # lr
        ax2.set_yscale('log')                
        lr1, lr2 = np.min(trn.lr), np.max(trn.lr)
        if lr1 > 0 and lr2 > 0:
            lr1, lr2 = math.floor(np.log10(lr1)), math.ceil(np.log10(lr2))
            if lr1 == lr2:
                lr1 -= 1; lr2 +=1
            lr1, lr2 = 10**lr1, 10**lr2
            ax2.set_ylim(lr1, lr2)

        x = np.array(trn.samples)/c_unit if unit=='sample' else np.array(trn.epochs)
        y = trn.lr
        if len(x) != len(y):        
            print(f"Plot warning: {kind}: lr {len(x)} != {len(y)}")
            x, y = x[:min(len(x),len(y))], y[:min(len(x),len(y))]
        ax2.plot(x, y, ":", color='darkred')                 
        ax2.legend(['lr'], loc='upper right', frameon = False)
        ax2.tick_params(axis='y', colors='darkred')
        if sub == 121:
            ax2.set_yticklabels([])   
            ax2.minorticks_off() # for log scale         


def plot_smooth_line(ax=None, x=[],y=[], color="black", label="", count=200, kern=51, stride=1, width=1.5, alpha=0.5, num=10):    
    if len(x) != len(y):        
        print(f"Plot warning: {len(x)} != {len(y)}")
        x, y = x[:min(len(x),len(y))], y[:min(len(x),len(y))]

    lw = 1.5 if len(x) < 100 else (1 if len(x) < 200 else 0.5)        
    if len(x) <  count:
        ax.plot(x, y, color, linewidth=lw, label=label)
    else:
        if alpha > 0:
            ax.plot(x, y, color, linewidth=lw, alpha=alpha)
        pool = nn.AvgPool1d(kern, stride=stride, padding=kern // 2, count_include_pad=False)
        avg_y = pool(torch.Tensor(y).view(1,-1)).flatten().numpy()
        avg_x = pool(torch.Tensor(x).view(1,-1)).flatten().numpy()
        
        sx, sxx = avg_x[-num:].sum(), (avg_x[-num:]*avg_x[-num:]).sum()
        sy, sxy = avg_y[-num:].sum(), (avg_x[-num:]*avg_y[-num:]).sum()
        d = num*sxx - sx*sx
        if d > 0:
            b = (num*sxy - sx*sy) / d                                 # экстраполируем вперёд
            a = (sy - b*sx) / num                                     # даже при stride=1 график отстаёт                  
            avg_x = np.append(avg_x, [ x[-1] ])                       # из-за усреднения x            
            avg_y = np.append(avg_y, [a + b*x[-1]])
                
        ax.plot(avg_x, avg_y, color, linewidth=width, label=label)    
        


#===============================================================================
#                                   Main
#===============================================================================

if __name__ == '__main__':
    x = np.random.normal(size=(1000,))
    plot_histogram(x, w=8, digits=3)
