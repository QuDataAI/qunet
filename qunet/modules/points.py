import copy
import torch, torch.nn as nn

from ..config   import Config
from .mlp      import MLP
#===============================================================================

class  PointsBlock(nn.Module):
    def __init__(self,  *args, **kvargs) -> None:
        super().__init__()
        self.cfg = PointsBlock.default()
        self.cfg.set(*args, **kvargs)
        self.create()

    def default():
        return copy.deepcopy(Config(
            E  = 64,          # размерность эмбедига
            max  = False,
            mean = True,
            res  = 1,           # residual петли (0-нет, 1-обычные, 2-тренеруемые)
            mlp1 = MLP.default(),
            mlp2 = MLP.default(),
        ))

    def create(self):
        cfg = self.cfg
        assert cfg.mean or cfg.max, f"PointsBlock need mean or/and max, cfg={cfg.get_str()}"
        self.cfg.mlp1(input = cfg.E, output = cfg.E)
        self.cfg.mlp2(input = cfg.E, output = cfg.E)

        E, E2 = cfg.E, cfg.E
        self.ln_1 = nn.LayerNorm(E)
        self.ln_2 = nn.LayerNorm(E)

        n_cat = 2 if cfg.max and cfg.mean else 1
        self.mlp_1 = MLP(input=E, stretch=cfg.mlp1.stretch, output=E2)
        self.fc_w  = nn.Linear(n_cat*E2, E*E)
        self.fc_b  = nn.Linear(n_cat*E2, E)
        self.mlp_2 = MLP(input=E, stretch=cfg.mlp1.stretch, output=E)

        if cfg.res == 2:
            self.w_1   = nn.Parameter( torch.ones(1) )
            self.w_2   = nn.Parameter( torch.ones(1) )
        else:
            self.w_1   = cfg.res
            self.w_2   = cfg.res

        self.E       = E

    def forward(self, x):                                       # (B,T, E)
        x = self.ln_1(x)                                        # (B,T, E)
        y = self.mlp_1(x)                                       # (B,T, E')
        agg = []
        if self.cfg.mean:
            agg.append( y.mean(dim=1) )
        if self.cfg.max:
            agg.append( y.max(dim=1)[0] )
        y = torch.cat(agg, dim=1)                               # (B, n_cat*E')
        w = self.fc_w(y).view(-1,self.E, self.E)                # (B, E*E) -> (B, E,E)
        b = self.fc_b(y)[:,None,:]                              # (B, E)   -> (B, 1,E)

        y = nn.functional.gelu(torch.bmm(x, w) + b)                      # (B,T,E) @ (B,E,E) + (B,1,E) = (B,T,E)
        y = y + x * self.w_1                                    # (B,T, E)
        #y = gelu(y)

        x = self.ln_2(y)                                        # (B,T, E)
        y = self.mlp_2(x)
        y = y  + x * self.w_2                                   # (B,T, E)
        #y = gelu(y)
        return y                                                # (B,T, E)

#===============================================================================
