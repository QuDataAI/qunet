import time, gc, psutil, os, random, torch, numpy as np

class Info:
    def __init__(self) -> None:
        self.beg  = time.time()
        self.last = time.time()

    def info(self, text, pref="", end="\n"):
        """ 
        Information about the progress of calculations (time and memory) 
        """
        gc.collect()
        ram, t = psutil.virtual_memory().used / 1024**3,  time.time()    
        print(f"{pref}{(t-self.beg)/60:6.1f}m[{t-self.last:+6.1f}s] {ram:6.3f}Gb | {text}", end = end)
        self.last = time.time(); 
        return self

    def reset(self):
        self.beg  = time.time()
        self.last = time.time()
        return self

    def __call__(self, text, pref="", end="\n"):
        self.info(text, pref=pref, end=end)

#===============================================================================

def set_seed(seed: int = 137, verbose=1) -> None:
    """ 
    Setting Up Random Seeds In PyTorch
    https://wandb.ai/sauravmaheshkar/RSNA-MICCAI/reports/How-to-Set-Random-Seeds-in-PyTorch-and-Tensorflow--VmlldzoxMDA2MDQy
    """
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)

    if verbose:
        print(f"Random seed set as {seed}")
