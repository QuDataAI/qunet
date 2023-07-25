import torch
from torch.optim import Optimizer
# Code from https://github.com/google/automl/tree/master/lion

class Lion(Optimizer):
    def __init__(self, params, lr=1e-4, betas=(0.9, 0.99), weight_decay=0.0, zero=True):
        """
        Args
        ------------
            zero (bool=True):
                first average grad equal 0 else equal grad
            
        """
        if not 0.0 <= lr:
            raise ValueError('Invalid learning rate: {}'.format(lr))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError('Invalid beta parameter at index 0: {}'.format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError('Invalid beta parameter at index 1: {}'.format(betas[1]))
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
        self.zero = zero
        super().__init__(params, defaults)
    
    
    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                p.data.mul_(1 - group['lr'] * group['weight_decay'])

                grad = p.grad
                state = self.state[p]
                if len(state) == 0:
                    if self.zero:
                        state['exp_avg'] = torch.zeros_like(p)
                    else:
                        state['exp_avg'] = grad

                exp_avg = state['exp_avg']
                beta1, beta2 = group['betas']

                update = exp_avg * beta1 + grad * (1 - beta1)
                p.add_(torch.sign(update), alpha = -group['lr'])
                exp_avg.mul_(beta2).add_(grad, alpha = 1 - beta2)
        
        return loss
