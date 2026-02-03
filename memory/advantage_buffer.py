import random
import numpy as np

class AdvantageBuffer:
    def __init__(self, capacity):
        self.capacity = capacity 
        self.buffer = []

    def add(self, info_set, advantage_values):
        """
        Adds a sample to the buffer.
        """
        if len(self.buffer) < self.capacity:
            self.buffer.append((info_set, advantage_values))
        else:
            # Standard replacement if capacity is met [cite: 106]
            idx = random.randint(0, self.capacity - 1)
            self.buffer[idx] = (info_set, advantage_values)

    def clear(self):
        """
        CRITICAL: VR-DeepPDCFR+ clears this buffer every iteration.
        """
        self.buffer = []

    def sample(self, batch_size):
        return random.sample(self.buffer, min(len(self.buffer), batch_size))