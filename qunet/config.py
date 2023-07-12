import copy
from .utils import Info

#===============================================================================

class Config:
    def __init__(self, *args, **kvargs):
        """
        Universal parameter store
        Example:
        ```
        cfg = Config(x=2, y=5, z=Config(v=0))
        print(cfg)         # x:2, y:5, z: {v:0, }, 
        cfg(x=5, y=2)      # x:5, y:2, z: {v:0, }, 
        cfg2 = Config(x=0)
        cfg(cfg2)          # x:0, y:2, z: {v:0, }, 
        ```
        """

        self.check_variable_existence = False
        self.set(*args, **kvargs)
        self.check_variable_existence = True

    def __call__(self, *args,  **kvargs):
        return self.set(*args, **kvargs)

    def __str__(self):
        return self.get_str()

    def get(self, param, default=None):
        """ get param value by name """
        return self.__dict__[param] if param in self.__dict__ else default

    def copy(self):
        return copy.deepcopy(self)

    def protect(self, check=True):
        """ check=True - protected version (checks for the presence of a variable), otherwise no """
        self.check_variable_existence = check

    def has(self, param):
        """ check if param exists in config """
        return param in self.__dict__

    def set(self, *args, **kvargs):
        for a in args:            
            assert isinstance(a, Config), "Config args can  be only another Config!"            
            self.set_cfg(a)

        for k, v in kvargs.items():            
            if self.check_variable_existence and k not in self.__dict__:
                print(f'! Config warning: no property "{k}" set({k}={v})')            
            if isinstance(v, Config):
                if k not in self.__dict__:
                    self.__dict__[k] = Config()
                self.__dict__[k].set_cfg(v)
            else:
                self.__dict__[k] = v
           
        return self
        
    def set_cfg(self, cfg):
        """ set values from another Config """
        for k,v in cfg.__dict__.items():            
            if isinstance(v, Config): 
                if k not in self.__dict__:
                    self.__dict__[k] = Config()
                self.__dict__[k].set_cfg(v)
            elif not k.startswith("__") and k not in ['check_variable_existence', 'set']:
                self.__dict__[k] = v

    def get_str(self, end=", ", exclude=[]):
        """ output of config parameters """
        res = ""
        for k,v in self.__dict__.items():
            if not k.startswith("__") and k not in ["check_variable_existence", "set"] + exclude:
                if isinstance(v, Config):
                    res += f"{k}:"+" {"+v.get_str(exclude=exclude)+"}"+end
                else:
                    if type(v) == str:
                        res +=  f"{k}:'{v}'{end}"
                    else:
                        res +=  f"{k}:{v}{end}"
        return res
    
    def get_jaml(self, exclude=[], level=0):
        """ output of config parameters """
        res = ""
        for k,v in self.__dict__.items():
            if not k.startswith("__") and k not in ["check_variable_existence", "set"] + exclude:
                if isinstance(v, Config):
                    res += " "*(4*level) + f"{k}:\n" +v.get_jaml(exclude=exclude, level=level+1)+"\n"
                else:
                    if type(v) == str:
                        res += " "*(4*level) + f"{k}: '{v}'\n"
                    else:
                        res +=   " "*(4*level) + f"{k}: {v}\n"
        return res    

    def get_dict(self, exclude=[]):
        """ output of config parameters """
        res = {}
        for k,v in self.__dict__.items():
            if not k.startswith("__") and k not in ["check_variable_existence"] + exclude:
                if isinstance(v, Config):
                    res[k] = v.get_dict(exclude=exclude)
                else:
                    res[k] = v
        return res


if __name__ == '__main__':    
    info = Info()
    info("begin")

    cfg1 = Config(x=1, y=2)
    cfg2 = Config(cfg1, z=3)
    print(cfg2)

    cfg1 = cfg2.copy()
    cfg1.z=8
    print(cfg2)

    cfg2.cfg = Config(a=1, b=Config(z=4, t=9))
    print(cfg2.get_dict())

    print(cfg2.get_jaml())

    if False:
        cfg = Config(x=2, y=5, z=Config(v=0))
        print(cfg)         # x:2, y:5, z: {v:0, }, 
        cfg(x=5, y=2)      # x:5, y:2, z: {v:0, }, 
        print(cfg) 
        cfg2 = Config(x=0)
        cfg(cfg2)          # x:0, y:2, z: {v:0, }, 
        print(cfg) 

    if False:
        print("---------")
        cfg = Config(
            x=3, 
            z=4, 
            loss=Config(z=5)
        )
        print('cfg: ', cfg)
        cfg.loss(x=8)
        print('cfg: ', cfg)
        cfg.loss.protect(False)
        cfg.loss(a='a')
        print('cfg: ', cfg)
        print('-----------------')
        cfg2 = Config( x=5 )
        cfg.set_cfg(cfg2)
        print(cfg)
