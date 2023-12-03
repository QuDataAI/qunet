import numpy as np
import torch

class MemoryBuffer:
    """
    Overwrite memory for storing the environment model.
    This implementation is not the most efficient, but it is needed for experimenting with sorting.
    """
    def __init__(self, capacity, n_states, n_actions):
        """
        capacity  - memory capacity (maximum number of stored items)
        n_states  - number of state variables
        n_actions - number of actions
        """
        self.capacity = capacity    # maximum number of stored elements
        self.count = 0              # how many items have already been saved
        self.index = 0              # index of element to be inserted
        self.nS = n_states          # dim of the state
        self.nA = n_actions         # number of actions

        self.memo = np.zeros( (capacity, self.nS*2 + 4), dtype = np.float32)
    #------------------------------------------------------------------------------------

    def add(self, s0, a0, s1, r1, done, rewrite = 0):
        """ Add example to memory """
        self.index = self.index % self.capacity

        norm = np.dot(s1,s1)**0.5
        self.memo[self.index] = np.concatenate(( s0, s1, [a0], [r1], [done], [norm]) )

        self.index += 1
        self.count += 1

        if  abs(rewrite) < 1.0 and (self.count == self.capacity
            or (self.count > self.capacity and self.index >= int(abs(rewrite)*self.capacity)) ):
            self.memo = self.memo[ self.memo[:, -1].argsort() ]
            if rewrite < 0:
                self.memo = self.memo[::-1]   # large norm at the beginig
            self.index = 0
    #------------------------------------------------------------------------------------

    def samples(self, count):
        """ Return count of random examples from memory """
        mem_max = min(self.count, self.capacity)
        indxs = np.random.choice(mem_max, count, replace=False)
        sample = self.memo[indxs]
        s0 = sample[:, 0:           self.nS]
        s1 = sample[:, self.nS:     2*self.nS]
        a0 = sample[:, 2*self.nS:   2*self.nS+1]
        r1 = sample[:, 2*self.nS+1: 2*self.nS+2]
        en = sample[:, 2*self.nS+2: 2*self.nS+3]
        return torch.tensor(s0), torch.tensor(a0, dtype = torch.int64), torch.tensor(s1), torch.tensor(r1), torch.tensor(en)
    #------------------------------------------------------------------------------------

    def stat(self):
        """ Statistic of s1 length and actions """
        num = min(self.count, self.capacity)
        if num == 0:
            return [],[],[],[]

        s1 = self.memo[:num, self.nS: 2*self.nS]
        hist_S, bins_S = np.histogram(s1, bins=np.linspace(0, np.sqrt(self.nS), 101), density=True)

        a = self.memo[:num, 2*self.nS: 2*self.nS+1],
        hist_A, bins_A = np.histogram(a, bins=np.linspace(-0.5, self.nA-0.5, self.nA+1), density=True)

        return hist_S, bins_S, hist_A, bins_A

