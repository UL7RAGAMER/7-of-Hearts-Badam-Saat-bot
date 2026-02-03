import random
import torch

class HistoryBuffer:
    def __init__(self, capacity=1000000):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def add(self, info_set, action_idx, utility):
        """
        Stores (state, action, reward) for the Q-network baseline[cite: 191, 192].
        """
        sample = (info_set, action_idx, utility)
        if len(self.buffer) < self.capacity:
            self.buffer.append(sample)
        else:
            self.buffer[self.position] = sample
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.buffer, min(len(self.buffer), batch_size))