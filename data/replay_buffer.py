import random
import numpy as np
import torch

class PrioritizedReplayBuffer:
    def __init__(self, capacity=100000, alpha=0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = []
        self.pos = 0
        self.priorities = np.zeros((capacity,), dtype=np.float32)

    # Updated signature to accept opp_hands
    def push(self, state, action, reward, next_state, done, opp_hands):
        max_prio = self.priorities.max() if self.buffer else 1.0
        
        # Store as 6-element tuple
        data = (state, action, reward, next_state, done, opp_hands)
        
        if len(self.buffer) < self.capacity:
            self.buffer.append(data)
        else:
            self.buffer[self.pos] = data
        
        self.priorities[self.pos] = max_prio
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size, beta=0.4):
        if len(self.buffer) == self.capacity:
            prios = self.priorities
        else:
            prios = self.priorities[:len(self.buffer)]

        probs = prios ** self.alpha
        probs /= probs.sum()

        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]

        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()
        weights = np.array(weights, dtype=np.float32)

        # Unpack 6 elements
        states, actions, rewards, next_states, dones, opp_hands = zip(*samples)
        
        return (
            states, actions, rewards, next_states, dones, opp_hands, 
            indices, 
            torch.from_numpy(weights)
        )

    def update_priorities(self, batch_indices, batch_priorities):
        for idx, prio in zip(batch_indices, batch_priorities):
            self.priorities[idx] = prio + 1e-6

    def __len__(self):
        return len(self.buffer)