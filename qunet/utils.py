
class Config:
    def __init__(self,**kwargs):
        self.check_variable_existence = False
        self.set(**kwargs)
        self.check_variable_existence = True

    def __call__(self,  **kwargs):
        return self.set(**kwargs)

    def __str__(self):
        return self.get()

    def protect(self, check=True):
        """ check=True - protected version (checks for the presence of a variable), otherwise no """
        self.check_variable_existence = check

    def set(self, **kwargs):
        for k, v in kwargs.items():            
            if self.check_variable_existence and k not in self.__dict__:
                print(f'! Config warning: no property "{k}" set({k}={v})')
            self.__dict__[k] = v
            #if isinstance(v, Config):
            #    self.__dict__[k](v)
            #else:
            #    self.__dict__[k] = v
           
        return self
        
    def get(self, end=", ", exclude=[]):
        """ output of config parameters """
        res = ""
        for k,v in self.__dict__.items():
            if not k.startswith("__") and k not in ["get", "check_variable_existence"] + exclude:
                if isinstance(v, Config):
                    res += f"{k}:"+" {"+v.get()+"}"+end
                else:
                    if type(v) == str:
                        res +=  f"{k}:'{v}'{end}"
                    else:
                        res +=  f"{k}:{v}{end}"
        return res


if __name__ == '__main__':    

    cfg = Config(
        x=3, 
        z=4, 
        loss=Config(z=5)
    )


    print('cfg: ', cfg)
    cfg.loss(x=8)
    print('cfg: ', cfg)
    #cfg.loss.protect(False)
    cfg.loss(a='a')
    print('cfg: ', cfg)

