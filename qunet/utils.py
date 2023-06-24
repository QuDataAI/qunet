import time, gc, psutil

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
