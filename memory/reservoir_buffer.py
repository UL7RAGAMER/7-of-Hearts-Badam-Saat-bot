import random


class ReservoirBuffer:
    def __init__(self, capacity):
        self.capacity = capacity 
        self.buffer = []
        self.seen_so_far = 0

    def add(self, info_set, iteration_t, strategy_probs):
        """
        Adds samples using the Reservoir Sampling algorithm[cite: 821, 1125].
        """
        self.seen_so_far += 1
        sample = (info_set, iteration_t, strategy_probs)
        
        if len(self.buffer) < self.capacity:
            self.buffer.append(sample)
        else:
            # Probability of keeping this sample decreases over time [cite: 1125]
            j = random.randint(0, self.seen_so_far - 1)
            if j < self.capacity:
                self.buffer[j] = sample

    def sample(self, batch_size):
        return random.sample(self.buffer, min(len(self.buffer), batch_size))