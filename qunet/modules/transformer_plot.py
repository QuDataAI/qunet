import numpy as np, matplotlib.pyplot as plt

#===============================================================================

def plot_transformer_blocks(blocks, w=12, h=3, eps=1e-8, bar_width = 0.25, info=False):
    
    idx = np.arange(len(blocks))

    fig, ax = plt.subplots(1,1, figsize=(w, h))

    ax.grid(ls=":")
    ax.set_xticks(idx)
    ax.set_yscale('log')
    ax.set_ylabel("dx/x");  ax.set_xlabel("blocks");

    plt.text(0,0,f" std\n", ha='left', transform = ax.transAxes, fontsize=6, family="monospace")
            
    fft_dv, att_dv, mlp_dv = [0]*len(blocks), [0]*len(blocks), [0]*len(blocks)
    for i,block in enumerate(blocks):
        for layer in block.layers:
            dx = (layer.sqr_dx / (layer.sqr_x+eps)).sqrt().cpu().item()
            if layer.name == 'fft':  fft_dv[i] = dx
            if layer.name == 'att':  att_dv[i] = dx
            if layer.name == 'mlp':  mlp_dv[i] = dx
    #ax.set_ylim(ymin=0, ymax=max(np.max(fft), np.max(att), np.max(mlp)*1.1) ) 
    
    ax.bar(idx,              fft_dv, width = bar_width, edgecolor ='grey', alpha=0.5)
    ax.bar(idx +  bar_width, att_dv, width = bar_width, edgecolor ='grey', alpha=0.5)
    ax.bar(idx +2*bar_width, mlp_dv, width = bar_width, edgecolor ='grey', alpha=0.5)

    
    ymin, ymax = ax.get_ylim()
    for i,block in enumerate(blocks):
        for layer in block.layers:
            st =  f"{layer.std.cpu().item():.1f}\n"
            if layer.name == "fft":
                plt.text(i,            ymin, st + "fft", ha='center', fontsize=6, family="monospace")
            if layer.name == "att":
                plt.text(i+bar_width,  ymin, st + "att", ha='center', fontsize=6, family="monospace")
            if layer.name == "mlp":
                plt.text(i+2*bar_width,  ymin, st + "mlp", ha='center', fontsize=6, family="monospace")


    ax2 = ax.twinx()
    fft_gamma_d, att_gamma_d, mlp_gamma_d = [0]*len(blocks), [0]*len(blocks), [0]*len(blocks)
    for i,block in enumerate(blocks):
        for layer in block.layers:
            d = layer.gamma_d.sqrt().cpu().item()
            if layer.name == 'fft':  fft_gamma_d[i] = d
            if layer.name == 'att':  att_gamma_d[i] = d
            if layer.name == 'mlp':  mlp_gamma_d[i] = d
    ax2.plot(idx,               fft_gamma_d, marker=".")
    ax2.plot(idx +   bar_width, att_gamma_d, marker=".")
    ax2.plot(idx + 2*bar_width, mlp_gamma_d, marker=".")
    ax2.set_ylabel("gamma")

    ax3 = ax.twinx()
    ax3.set_yscale('log')
    fft_gamma_g, att_gamma_g, mlp_gamma_g = [0]*len(blocks), [0]*len(blocks), [0]*len(blocks)
    for i,block in enumerate(blocks):
        for layer in block.layers:
            g = layer.gamma_g.sqrt().cpu().item()
            if layer.name == 'fft':  fft_gamma_g[i] = g
            if layer.name == 'att':  att_gamma_g[i] = g
            if layer.name == 'mlp':  mlp_gamma_g[i] = g
    ax3.plot(idx,               fft_gamma_g, ":", marker=".")
    ax3.plot(idx +   bar_width, att_gamma_g, ":", marker=".")
    ax3.plot(idx + 2*bar_width, mlp_gamma_g, ":", marker=".")        
    ax3.set_ylabel("gamma grad")
    ax3.spines["right"].set_position(("outward", 50))    

    plt.show()

    if info:
        return {
            'fft_dv': fft_dv, 'fft_gamma_d': fft_gamma_d, 'fft_gamma_g': fft_gamma_g,
            'att_dv': att_dv, 'att_gamma_d': att_gamma_d, 'att_gamma_g': att_gamma_g,
            'mlp_dv': mlp_dv, 'mlp_gamma_d': mlp_gamma_d, 'mlp_gamma_g': mlp_gamma_g,
        }
