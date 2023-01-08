import torch
import random

class Buffer:
    def __init__(self, args):
        self.args = args
        self.data_pool = []
        self.p_index = -1
    
    def add(self, state):
        for i in range(self.args.round_size):
            self.p_index = (self.p_index + 1) % self.args.buffer_size
            if len(self.data_pool) < self.args.buffer_size:
                self.data_pool.append(state[i,:].clone().detach())
            else:
                self.data_pool[self.p_index] = state[i,:].clone().detach()
    
    def sample(self):
        if len(self.data_pool) < self.args.batch_size:
            return []
        index = [random.randint(0, len(self.data_pool) - 1) for _ in range(self.args.batch_size)]
        return torch.cat([self.data_pool[i].unsqueeze(0) for i in index], 0)
