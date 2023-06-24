import numpy as np, matplotlib.pyplot as plt
import torch, torch.nn as nn

class ModelState:
    def __init__(self, model, beta=0.8):
        """
        Example
        ------------
        ```
            state = ModelState(model)

            state.num_params() # number of model parameters
            
            state.layers()     # model layers
            state.params()     # model parameters 
            state.state()      # params and register_buffer

            state.plot()       # draw params and grads
            ModelState.hist_params([m.weight, m.bias], ["w","b"])
        ```
        """
        self.model = model
        self.beta  = min(0.999, max(0.001, beta))
        self.__params   = {}
        self.__layers   = []
        self.__layersID = {}
        self.agg        = 0

    #---------------------------------------------------------------------------

    @staticmethod
    def i2s(v, width=10):
        fmt = "d"
        v_st  = f"{v:,{fmt}}"  if v > 0 else ""
        return " "*(width-len(v_st)) + v_st

    #---------------------------------------------------------------------------

    def num_params(self, grad=True):
        """
        Return number of model parameters

        Args:
        ------------
            grad (bool=True):
                parameters with gradient only (True), without gradient (False), all (None)
        """
        if grad == True:
            return sum([param.numel() for param in self.model.parameters() if param.requires_grad])
        if grad == False:
            return sum([param.numel() for param in self.model.parameters() if not param.requires_grad])
        return sum([param.numel() for param in self.model.parameters() ])


    #---------------------------------------------------------------------------

    def reset(self):
        """
        Set initial statistics values for each model parameter
        """
        self.__params = dict()
        for n, p in self.model.named_parameters():
            self.__params[n] = {
                'numel': p.numel(),
                'is_grad': p.requires_grad,
                'shape': tuple(p.shape),
                'data' : torch.square(p.data).sum().cpu(),
                'min'  : p.data.abs().min().cpu(),
                'max'  : p.data.abs().max().cpu(),
                'grad' : 0,
            }
            if p.grad is not None:
                self.__params[n]['grad'] = torch.square(p.grad).sum().cpu()

    #---------------------------------------------------------------------------

    def update(self):
        """
        Accumulate averages of statistics using exponential average
        """
        model = self.model
        if len(self.__params) == 0:
            self.reset()
            return

        w1, w2 = 1-self.beta, self.beta
        for n, p in model.named_parameters():
            param = self.__params[n]
            param['data'] = w1 * param['data'] + w2 * torch.square(p.data).sum().cpu()
            param['min']  = w1 * param['min']  + w2 * p.data.abs().min().cpu()
            param['max']  = w1 * param['max']  + w2 * p.data.abs().max().cpu()
            if p.grad is not None:
                if 'grad' not in param:
                    param['grad'] = torch.square(p.grad).sum().cpu()
                else:
                    param['grad'] = w1 * param['grad'] + w2 * torch.square(p.grad).sum().cpu()

    #---------------------------------------------------------------------------

    def sum_values(self, params, kind='numel'):
        tot = 0
        for n in params:
            tot += self.__params[n][kind]
        return tot

    #---------------------------------------------------------------------------

    def get_groups(self, agg=0):
        """
        Create parameter group dictionaries (to aggregate statistics)

        Args:
        ------------
            agg (int=None):
                cut off the agg of the last levels of the parameter name to aggregate them (level0.weight -> level0)
        """
        if agg is None:
            agg = self.agg
        else:
            self.agg = agg

        groups, names = dict(), dict()
        for n, _ in self.model.named_parameters():
            parts = n.split(".")
            parts = parts if agg==0 else parts[:- min(len(parts)-1, agg)]
            base  = ".".join(parts)
            if base in groups:
                groups[base]['names'].append(n)
            else:
                groups[base] = { 'names': [n], 'numel':[], 'data':[], 'min':[], 'max':[], 'shape':[], 'grad': [], 'is_grad': [] }
            names[n] = base
        return groups, names

    #---------------------------------------------------------------------------

    def groups_init(self, groups):
        for group in groups.values():
            group['numel'] = []; group['shape']  = []; group['data']=[];
            group['min']   = []; group['max']  = [];
            group['grad']  = []
            group['is_grad'] = []

    #---------------------------------------------------------------------------

    def aggregate(self, agg=None):
        """
        """
        groups, names = self.get_groups(agg=agg)
        self.groups_init(groups)
        if len(self.__params) == 0:
            self.reset()

        for name, param in self.__params.items():
            group = names[name]
            groups[group]['numel'].append(param['numel'])
            groups[group]['data'] .append( param['data'] )
            groups[group]['grad'] .append( param['grad'] )
            groups[group]['min']  .append( param['min'])
            groups[group]['max']  .append( param['max'])
            groups[group]['shape'].append( param['shape'])
            groups[group]['is_grad'].append( param['is_grad'])

        for group in groups.values():
            numel = group['numel'] = sum(group['numel'])
            group['data']  = (sum(group['data']) / numel)**0.5  if numel > 0 else 0
            group['min'] = min(group['min'])
            group['max'] = min(group['max'])
            group['shape'] = group['shape'][-1]         # ?
            group['is_grad'] = sum(group['is_grad']) / len(group['is_grad'])
            if len(group['grad']):
                group['grad'] = (sum(group['grad']) / numel)**0.5  if numel > 0 else 0
            else:
                group['grad'] = 0

        return groups

    #---------------------------------------------------------------------------

    def get_layers(self, model, depth=0, name="",  layers=[]):
        for i, (nm,mo) in enumerate(model.named_children()):
            n_kids = len(list(mo.children()))
            if n_kids > 0:
                self.__layersID[id(mo)] = len(layers)
                layers.append({'name': name+"."+nm,
                              'params':[], 'depth': depth, 'i':i, 'n_kids': n_kids, 'module': mo,
                              'input':'', 'output':''})
                self.get_layers(mo, depth=depth+1, name=name+"."+nm if len(name) else nm, layers=layers)
            else:
                self.__layersID[id(mo)] = len(layers)
                layers.append({'name': name+"."+nm,
                              'params':[name +"." + nm + "."+ n if len(name) else nm+"."+n for n,_ in mo.named_parameters()],
                              'depth': depth, 'i':i, 'n_kids': 0, 'module': mo,
                              'input':'', 'output':'' })

    #---------------------------------------------------------------------------

    def get_layers_descr(self, info = 0):
        for lr in self.__layers:
            mo, depth, i, count = lr['module'], lr['depth'], lr['i'], lr['n_kids']
            nm = lr['name'].split('.')[-1]
            if count:
                lr['st']=f"{'│' if depth else ''}{' '*max(0,3*depth-1)+'└─ ' if depth else '├─ '}{mo.__class__.__name__ if info >= 0 else nm}"
            else:
                layer = mo.__class__.__name__
                descr = layer if info >= 0 else nm
                if info == 1:
                    if   layer == "Linear":   descr += f"({mo.in_features}->{mo.out_features})"
                    elif layer == "Bilinear": descr += f"({mo.in1_features},{mo.in2_features}->{mo.out_features})"
                    elif layer in ["Dropout", "Dropout2d", "Dropout3d"]:   descr += f"({mo.p})"
                    elif layer in ["Conv1d","Conv2d","Conv3d"]: descr += f"({mo.in_channels}->{mo.out_channels})"
                    elif layer in ["MaxPool1d", "MaxPool2d", "MaxPool3d"]: descr += f"({mo.kernel_size})"
                    elif layer == 'Embedding': descr += f"({mo.num_embeddings},{mo.embedding_dim})"
                    elif layer in ['RNN','GRU','LSTM']: descr += f"({mo.input_size},{mo.hidden_size})"
                elif info > 1:
                    if   layer == "Linear": descr += f"({mo.in_features}->{mo.out_features}, {'T' if mo.bias is not None else 'F'})"
                    elif layer in ["Dropout", "Dropout2d", "Dropout3d"]:   descr += f"({mo.p})"
                    elif layer in ["Conv1d","Conv2d","Conv3d"]: descr += f"({mo.in_channels}->{mo.out_channels}, k:{mo.kernel_size}, s:{mo.stride}, p:{mo.padding}, {'T' if mo.bias is not None else 'F'})"
                    elif layer in ["MaxPool1d", "MaxPool2d", "MaxPool3d"]: descr += f"(k:{mo.kernel_size}, s:{mo.stride}, p:{mo.padding})"
                    elif layer in ["BatchNorm1d","BatchNorm2d","BatchNorm3d"]: descr += f"({mo.num_features})"
                    elif layer == "LayerNorm": descr += f"({mo.normalized_shape})"
                    elif layer == 'Embedding': descr += f"({mo.num_embeddings},{mo.embedding_dim})"
                    elif layer in ['RNN','GRU','LSTM']: descr += f"({mo.input_size},{mo.hidden_size}, l:{mo.num_layers}, bi:{'T' if mo.bidirectional else 'F'})"

                skip = f"│{' '*max(0,3*depth-1)}{'├─ ' if i+1 < count else '└─ '}" if depth > 0 else "├─ "
                lr['st'] = f"{skip}{descr}"

    #---------------------------------------------------------------------------
    #
    def layers(self, info=1, is_names=True, input_size=None, input_data=None):
        """
        Display information about model layers

        Args:
        ------------
            info (int=1):
                (-1): name of layer; (0,1,2) - name of layer class with parameters of varying degrees of detail
            is_name (bool=True):
                show path to given layer (for -1,0,1)
        """
        if len(self.__layers) == 0:
            self.get_layers(self.model, layers=self.__layers)
        self.get_layers_descr(info = info)                        # info can change

        if len(self.__layers) == 0:
            print(self.model.__class__.__name__ + " model empty")
            return

        if len(self.__params) == 0:
            self.reset()

        if input_size is not None or  input_data is not None:
            self.forward(input_size=input_size, input_data=input_data)
            inp_w = max( [len(ln['input'])  for ln in self.__layers ])
            out_w = max( [len(ln['output']) for ln in self.__layers ])

        total_num = self.num_params(True)
        ma = max( [ len(ln['st']) for ln in self.__layers ] )
        print(self.model.__class__.__name__ + " "*(ma-len(self.model.__class__.__name__))+"      params           data")
        for ln in self.__layers:
            shapes = ""
            if input_size is not None or  input_data is not None:
                shapes  = ln['input']  +' '*(inp_w-len(ln['input'])) + ' -> '
                shapes += ln['output'] +' '*(out_w-len(ln['output']))
                data = ""

            name = ""
            if is_names and info <= 1:
                lst = ln['name'].split('.')
                name = ".".join([ f"[{n}]" if n.isdigit() else n for n in lst])
                name = name.replace(".[", "[" )
                if name[0]==".":
                    name = name[1:]
                name = ' <  ' + name

            if len(ln['params']) > 0:
                num  = self.sum_values(ln['params'], kind='numel')
                prs  = 100*num/total_num
                data = f"{(self.sum_values(ln['params'], kind='data') / num)**0.5:5.3f}"
                num_st  = ModelState.i2s(num)
                prs_st  = f"~ {prs:3.0f}%" if prs > 0.5 else " "*6

                if input_size is not None or  input_data is not None:
                    data = ""

                print(f"{ln['st']+' '*(ma-len(ln['st']))}  {num_st}  {prs_st} | {data} {shapes} {name}")
            else:
                w = 19 if  input_size is not None or  input_data is not None else 24
                print(f"{ln['st']+' '*(ma-len(ln['st']))} {' '*w}    {shapes} {name}")

        print("="*(ma+12))
        n1, n2, n3 = self.num_params(True),  self.num_params(False),  self.num_params(None)
        print(f"{'trainable:'+' '*(ma-15)}     {ModelState.i2s(n1,12)}")
        if n2 or n3 != n1:
            print(f"{'other:'+' '*(ma-11)}     {ModelState.i2s(n2,12)}")
            print(f"{'total:'+' '*(ma-11)}     {ModelState.i2s(n3,12)}")

    #---------------------------------------------------------------------------

    def state(self):
        """
        Display information about all parameters (including register buffers)
        """
        ma = max([len(k)  for k in self.model.state_dict().keys()])
        descr = "param" + " "*(ma-5) + "   value            num  shape"
        print(descr)
        for k,v in self.model.state_dict().items():
            val = v if v.numel() < 2 else torch.sqrt(torch.square(v).mean())
            print(f"{k+' '*(ma-len(k))} {val.item():8.4f}  {ModelState.i2s(v.numel(),12)}  {tuple(v.shape)}")

    #---------------------------------------------------------------------------

    def forward_pre_hook(self, m, inputs):
        """
        Allows for examination and modification of the input before the forward pass.
        Note that inputs are always wrapped in a tuple.
        """
        assert id(m) in self.__layersID, f"module {id(m)} should be in __layersID"
        idx = self.__layersID[id(m)]
        if len(inputs):
            if torch.is_tensor(inputs[0]):
                self.__layers[idx]['input'] = f"{tuple(inputs[0].shape)}"
            else:
                self.__layers[idx]['input'] = "???"
        else:
            self.__layers[idx]['input'] = "None"

        #print(f"forward_pre_hook: {m.__class__.__name__}   {len(inputs)}   {self.__layers[idx]['input']}")

    #---------------------------------------------------------------------------

    def forward_hook(self, m, inputs, output):
        """
        Allows for examination of inputs / outputs and modification of the outputs
        after the forward pass. Note that inputs are always wrapped in a tuple while outputs are passed as-is.
        """
        assert id(m) in self.__layersID, f"module {id(m)} should be in __layersID"
        idx = self.__layersID[id(m)]
        if torch.is_tensor(output):
            self.__layers[idx]['output'] = f"{tuple(output.shape)}"
        elif type(output) in [list, tuple] and len(output) and torch.is_tensor(output[0]):
            self.__layers[idx]['output'] = f"{tuple(output[0].shape)}, ..."
        else:
            self.__layers[idx]['output'] = "???"

        #print(f"forward_hook   : {m.__class__.__name__}   {self.__layers[idx]['input']} -> {self.__layers[idx]['output']}")

    #---------------------------------------------------------------------------

    def forward(self, input_size=None, input_data=None):
        if len(self.__layers) == 0:
            self.get_layers(self.model, layers=self.__layers)
        for lr in self.__layers:
            lr['forward_pre_hook_handle'] = lr['module'].register_forward_pre_hook(self.forward_pre_hook)
            lr['forward_hook_handle']     = lr['module'].register_forward_hook    (self.forward_hook)

        x = None
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if input_data is not None:
            x = input_data
            if type(x) in [list, tuple]:
                for i in range(len(x)):
                    assert torch.is_tensor(x[i]), f"input_data should be torch tensor or tuple of tensors, got {type(x[i])}"
                    x[i] = x[i].to(device)
            else:
                assert torch.is_tensor(x), f"input_data should be torch tensor or tuple of tensors, got {type(x)}"
                x = x.to(device)

        elif input_size is not None:
            assert type(input_size) in [list, tuple], f"input_shape should be tuple or list, but got {type(input_size)}"
            x = torch.randn(tuple(input_size), device=device)

        if x is not None:
            self.model.to(device)
            with torch.no_grad():
                if type(x) in [tuple, list]:
                    y = self.model(*x)
                else:
                    y = self.model(x)

            print(f"{self.model.__class__.__name__}:  {tuple(x.shape)} -> {tuple(y.shape)}")

        for lr in self.__layers:
            if 'forward_pre_hook_handle' in lr:
                lr['forward_pre_hook_handle'].remove()
            if 'forward_hook_handle' in lr:
                lr['forward_hook_handle'].remove()

    #---------------------------------------------------------------------------

    def params(self, agg=None):
        """
        Display information about model parameters

        Args:
        ------------
            agg (int=None):
                cut off the agg of the last levels of the parameter name to aggregate them (level0.weight -> level0)
        """
        groups = self.aggregate(agg=agg)
        num = self.num_params(True)
        w = max([len(n) for n in groups.keys() ])
        print(f"  # {' '*w}      params          |mean|  [     min,      max ]  |grad|   shape")
        print("-"*(w+50))
        for i, (name, group) in enumerate(groups.items()):
            nm = name + " "*(w-len(name))
            prc = 100*group['numel']/num
            prc = f"~{prc:3.0f}%" if prc > 0.5 else "     "
            grad = f"{group['grad']:8.1e}" if group['is_grad'] > 0 else "-"*8
            print(f"{i:3d}: {nm}  {ModelState.i2s(group['numel'],9)} {prc}  {group['data']:8.3f}  [{group['min']:8.3f}, {group['max']:8.3f}]  {grad}  {group['shape']}  ")

        print("="*(w+12+4))
        n1, n2, n3 = self.num_params(True),  self.num_params(False),  self.num_params(None)
        print(f"{'trainable:'+' '*(w-15+4)}     {ModelState.i2s(n1,12)}")
        if n2 or n3 != n1:
            print(f"{'other:'+' '*(w-11+4)}     {ModelState.i2s(n2,12)}")
            print(f"{'total:'+' '*(w-11+4)}     {ModelState.i2s(n3,12)}")

        return groups

    #---------------------------------------------------------------------------

    def plot(self, agg=None, data=True, grad=True, alpha=0.5, w=12, h=3):
        """
        Draw a graph of the number of parameters, the average value of their modulus and the average value of the modulus of the gradient

        Args:
        ------------
            agg (int=None):
                cut off the agg of the last levels of the parameter name to aggregate them (level0.weight -> level0)
            data (bool=True):
                show average absolute value of parameters
            grad (bool=True):
                show average absolute value of gradients
            alpha (float=0.5):
                transparency for bars of the number of elements in the parameter tensor
            w,h (int):
                chart width and height

        """
        groups = self.aggregate(agg=agg)
        numel, data, mi, ma, grad, names = [], [], [], [], [], []
        for name, group in groups.items():
            numel.append(group['numel'])
            data.append(group['data'])
            mi.append(group['min'])
            ma.append(group['max'])
            grad.append(group['grad'])
            names.append(name)
        if len(data) <= 5:
            x = names
        else:
            x = np.arange(len(data))

        fig, ax = plt.subplots(1,1, figsize=(w, h))
        n1, n2 = self.num_params(), self.num_params(True)
        plt.title(f"params: {ModelState.i2s(n2)} ({100*n2/n1:.2f}%)")

        ax.set_yscale('log')
        ax.bar(x, numel, label="num", alpha=alpha, color='lightgray')
        ax.grid(ls=":")

        if data:
            ax1 = ax.twinx()            
            ax1.plot(x, data,   "-b.",  label="data")
            ax1.set_ylim(bottom=0)   # after plot !!!
            #ax1.errorbar(x, data,   yerr=[mi, ma],  fmt='-b.', elinewidth=2, capsize=4, lw=1)

            ax1.spines["left"].set_position(("outward", 30))
            ax1.spines["left"].set_visible(True)
            ax1.yaxis.set_label_position('left')
            ax1.yaxis.set_ticks_position('left')
            ax1.set_ylabel("data", color='b')
            ax1.tick_params(axis='y', colors='b')

        if grad:
            ax2 = ax.twinx()            
            ax2.plot(x, grad, "-r.", label="grad")
            ax2.set_ylim(bottom=0)   # after plot !!!
            ax2.set_ylabel("grad",  color='r')
            ax2.tick_params(axis='y', colors='r')

        plt.show()

    #---------------------------------------------------------------------------

    @staticmethod
    def plot_params(params, titles=None, sorted=1, grad=True, w=12, h=3):
        if type(params) not in [list, tuple]: params = [params]
        if type(titles) is str:               titles = [titles]
        fig, axs = plt.subplots(1,len(params), figsize=(w, h))
        for i, p in enumerate(params):
            ax = axs[i] if len(params) > 1 else axs
            if p.dim()==1 or (p.dim()==2 and (p.shape[0]==1 or p.shape[-1]==1)):
                if titles is not None and type(titles) in [list, tuple] and len(titles) > i:
                    ax.set_title(titles[i])
                ax.grid(ls=":")
                ax.set_ylabel("data",  color='b')
                ax.tick_params(axis='y', colors='b')
                y = p.data.flatten().cpu().numpy()
                if sorted < 2:                    
                    ax.plot(y, "-b", lw=0.5 if len(y) > 1000 else 1)
                if sorted >= 1:
                    y = np.sort(y)
                    ax.plot(y, "-g")

                if grad and p.grad is not None:
                    ax2 = ax.twinx()
                    ax2.set_ylabel("grad",  color='r')
                    g = p.grad.flatten().cpu().numpy()
                    g = np.sort(g)
                    ax2.plot(g, ":r")
                    ax2.tick_params(axis='y', colors='r')

            elif p.dim()==2:
                y = p.data.cpu().numpy()
                if sorted:
                    y = np.sort(y)
                ax.imshow(y)
        plt.show()

    #---------------------------------------------------------------------------

    @staticmethod
    def hist_params(params, titles=None, bins=50, w=12, h=3,  digits=2):        
        if type(params) not in [list, tuple]: params = [params]
        if type(titles) is str:               titles = [titles]

        fig, axs = plt.subplots(1,len(params), figsize=(w, h))
        for i, p in enumerate(params):
            ax = axs[i] if len(params) > 1 else axs                   
            values = p.data.flatten().cpu().numpy()
            #values = p.data.abs().amax(dim=0).flatten().cpu().numpy()
            #print(np.min(values), np.max(values))
            title = titles[i] if titles is not None and type(titles) in [list, tuple] and len(titles) > i else ""

            ModelState.hist_param(ax, values, title, bins=bins, digits=digits) 

        plt.show()
    
    #---------------------------------------------------------------------------

    @staticmethod
    def hist_param(ax, v, title, bins=50, digits=2):
        r = lambda x: '{x:.{digits}f}'.format(x=round(x,digits), digits=digits)                       

        y, x = np.histogram(np.abs(v), bins=bins*10, density=True)
        y = np.array([y[i]*(x[i+1]-x[i]) for i in range(len(y))])
        y = np.cumsum(y)
        x = np.array([0.5*(x[i]+x[i+1]) for i in range(len(x)-1)])
        ax.plot( x, y, "-g")
        ax.grid(ls=":"); ax.set_ylabel("sum prob |p|", color='g'); ax.set_xlabel("|p|", color='g')
        ax.set_title(f"{title} mean={r(v.mean())} ± {r(v.std())} [{r(v.min())}, {r(v.max())}]; cnt={len(v)}", fontsize=10)                
            
        ax2  = ax.twinx().twiny(); 
        ax2.hist( v, bins=bins, color="lightblue", ec="black", alpha=0.5)
        ax2.set_ylabel("N(p)", color="lightblue")


"""
Полный набор параметров, зарегистрированных модулем, можно просмотреть с помощью вызова parameters()
или named_parameters(), где последний включает имя каждого параметра.
Вызовы parameters()и named_parameters()будут рекурсивно включать все дочерние параметры.

Метод parameters() - это генератор только по обучаемым параметрам (его мы передаём оптимизатору).
Метод named_parameters() - аналогичный генератор, но дополнительно содержащий имена параметров.
Эти два метода позволяют, в т.ч., достучаться до градиентов параметров.
Кроме этого, есть словарь state_dict(), который обычно используется,
когда модель сохраняется в файле для последующей загрузки.
В нём присутствуют только данные и нет информации о градиентах, однако параметры есть все, включая не обучаемые.
"""
