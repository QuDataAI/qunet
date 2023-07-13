from inspect import isfunction
import torch, torch.nn as nn
import torch.nn.functional as F

#===============================================================================
#  Общие модули, использующиеся в различных классах папаки modules
#===============================================================================

class LayerNormChannels(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.norm = nn.LayerNorm(channels)
    #---------------------------------------------------------------------------

    def forward(self, x):           # (B,C,H,W,...)
        x = x.transpose(1, -1)      # (B,W,H,...,C)
        x = self.norm(x)            # avg on index c, 2*C parameters
        x = x.transpose(-1, 1)      # (B,C,H,W,...)
        return x

#===============================================================================

class ShiftFeatures(nn.Module):
    def __init__(self, std=0.):
        super().__init__()
        self.std = std
    #---------------------------------------------------------------------------        

    def forward(self, x):           # (B,E) or (B,T,E) or (B,E,H,W) or (B,E,D,H,W)
        if self.std == 0.0 or not self.training:
            return x
        
        if x.dim() == 2:
            B,E = x.shape
            return x + (self.std / E**0.5) * torch.randn((B,E), device=x.device)
        if x.dim() == 3:
            B,_,E = x.shape
            return x + (self.std / E**0.5) * torch.randn((B,1,E), device=x.device)
        if x.dim() == 4:
            B,C,_,_ = x.shape
            return x + (self.std / C**0.5) * torch.randn((B,C,1,1), device=x.device)
        if x.dim() == 5:
            B,C,_,_,_ = x.shape
            return x + (self.std / C**0.5) * torch.randn((B,C,1,1,1), device=x.device)
                
        assert False, f"Wrong dim of tensor x: {x.shape}"

#===============================================================================

class Scaler(nn.Module):
    def __init__(self, kind=0, value=1., dim=1,  E=None):
        super().__init__()        
        self.kind = kind
        if   kind == 2:                # training multiplier for each components
            if dim == 2:
                self.scale = nn.Parameter( torch.empty(1, E, 1,1).fill_(float(value)) )
            else:
                self.scale = nn.Parameter( torch.empty(1, 1, E ).fill_(float(value)) )

        elif kind == 1:                # training common multiplier
            self.scale = nn.Parameter( torch.tensor(float(value)) )     # 0 ??? 
        else:                         # constant multiplayer
            self.scale = None
    #---------------------------------------------------------------------------        

    def forward(self, x):        
        if self.scale is None:
            return x
        return x*self.scale


#===============================================================================
# Набор статических методов, создающих различные элементарные модули
#===============================================================================

class Create:

    @staticmethod
    def activation(fun, inplace=True):
        """
        Create activation layer from string/function.

        Args
        ----------
        activation (function, or str, or nn.Module):
            Activation function or name of activation function.
        inplace (bool: True)

        Returns
        -------
        nn.Module
            Activation layer.
        """
        assert (fun is not None)
        if isfunction(fun):
            return fun()
        elif isinstance(fun, str):
            if fun     == "sigmoid":
                return  nn.Sigmoid()
            elif fun   == "tanh":
                return nn.Tanh()
            elif fun   == "gelu":
                return  nn.GELU()    
            elif fun   == "relu":
                return nn.ReLU(inplace=inplace)
            elif fun == "relu6":
                return nn.ReLU6(inplace=inplace)
            elif fun == "swish":                 # https://arxiv.org/abs/1710.05941
                return  lambda x: x * torch.sigmoid(x)                      
            elif fun == "hswish":                # https://arxiv.org/abs/1905.02244
                return lambda x: x * F.relu6(x+3.0, inplace=inplace) / 6.0  
            elif fun == "hsigmoid":              # https://arxiv.org/abs/1905.02244
                return lambda x: F.relu6(x+3.0, inplace=inplace) / 6.0      
        
            else:
                raise NotImplementedError()
        else:
            assert isinstance(fun, nn.Module)
            return fun

    #---------------------------------------------------------------------------
    @staticmethod
    def norm(E,  norm, dim):
        if norm == 1:
            return nn.BatchNorm1d(E)    if dim==1 else nn.BatchNorm2d(E) 
        if norm == 2:
            return nn.LayerNorm (E)     if dim==1 else LayerNormChannels(E)
        if norm == 3:
            return nn.InstanceNorm1d(E) if dim==1 else nn.InstanceNorm2d(E)
        return nn.Identity()

    #---------------------------------------------------------------------------
    @staticmethod
    def dropout(dim, p=0):
        if dim == 1:
            return nn.Dropout(p)
        if dim == 2:
            return nn.Dropout2d(p)        
        return nn.Identity()

#===============================================================================
# Набор статических методов, изменяющих параметры модели
#===============================================================================

class Change:

    @staticmethod
    def linear(model):
        """Reset Conv2d, Linear weights to kaiming_normal_ and bias to 0
        """
        def init(m):
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None: nn.init.zeros_(m.bias)
        
        model.apply(init); 
    #---------------------------------------------------------------------------

    @staticmethod
    def dropout(model, p=0., info=False):
        """
        Set dropout rates of DropoutXd to value p. It may be float or list of floats.
        If there are more such modules than the length of the list p, the last value repeated.

        Example:
        ----------        
        Change.dropout(model, 2.0 )
        Change.dropout(model, [2.0, 3.0] )     
        """
        kind = (nn.Dropout, nn.Dropout1d, nn.Dropout2d, nn.Dropout3d)
        layers = Change.get_layers(model, kind=kind)
        if layers:
            if type(p) in (int, float):
                p = [p]

            for i,layer in enumerate(layers):
                layer.p = p[ min(i, len(p)-1) ]

                if info:
                    print(layer)
        else:
            if info:
                print(f"No layers {kind}")

    #---------------------------------------------------------------------------

    @staticmethod
    def shift(model, std=0., info=False):
        """
        Set std in ShiftFeatures. It may be float or list of floats.
        If there are more such modules than the length of the list std, the last value repeated.

        Example:
        ----------
        Change.set_shift(model, 2.0 )
        Change.set_shift(model, [2.0, 3.0] )
        """
        kind = (ShiftFeatures)
        layers = Change.get_layers(model, kind=kind)
        if layers:
            if type(std) in (int, float):
                std = [std]

            for i,layer in enumerate(layers):
                layer.std = std[ min(i, len(std)-1) ]

                if info:
                    print(layer)
        else:
            if info:
                print(f"No layers  {kind}")

    #---------------------------------------------------------------------------

    @staticmethod
    def drop_block(self, model, p=0.):
        if type(p) in (int, float):
            p = [p]

        layers = Change.get_layers_with_method(model, "set_drop_block")

        for i, layer in enumerate(layers):            
            layer.set_drop_block( p[ min(i, len(p)-1) ] )

    #---------------------------------------------------------------------------

    @staticmethod
    def get_layers(model, kind):        
        """Get a list of model layers of type kind

        Example:
        ----------
        layers = get_model_layers(model, (nn.Dropout1d, nn.Dropout2d) )
        """
        layers = []
        Change.get_layers_rec(model, kind, layers)
        return layers

    #---------------------------------------------------------------------------

    @staticmethod
    def get_layers_rec(model, kind, layers):    
        for mo in model.children():
            if isinstance(mo, kind):
                layers.append(mo)
            else:                
                Change.get_layers_rec(mo, kind, layers) 

    #---------------------------------------------------------------------------

    @staticmethod
    def get_layers_with_method(model, method):        
        """Get a list of model layers wich has method

        Example:
        ----------
        layers = get_layers_with_method(model, 'forward' )
        """
        layers = []
        Change.get_layers_with_method_rec(model, method, layers)
        return layers

    #---------------------------------------------------------------------------

    @staticmethod
    def get_layers_with_method_rec(model, method, layers):    
        for mo in model.children():
            if hasattr(mo, method):
                layers.append(mo)
            else:                
                Change.get_layers_with_method_rec(mo, method, layers) 

