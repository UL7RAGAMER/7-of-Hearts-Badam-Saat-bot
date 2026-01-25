import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim)
        )

    def forward(self, x):
        return F.relu(x + self.net(x))

class BadaamSathDuelingDQN(nn.Module):
    def __init__(self, input_dim=119, output_dim=53):
        super().__init__()
        
        # 1. Shared Feature Extractor
        # This part now learns both Strategy AND Card Counting
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU()
        )
        
        self.res_blocks = nn.Sequential(
            ResidualBlock(512),
            ResidualBlock(512),
            ResidualBlock(512) # Added one more block for capacity
        )
        
        # 2. State Value stream (V)
        self.value_stream = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        
        # 3. Advantage stream (A)
        self.advantage_stream = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim)
        )

        # 4. NEW: Opponent Prediction Stream (Auxiliary Task)
        # Predicts probability of each of the 52 cards being held by opponents.
        # Output: 52 cards * 3 opponents = 156 logits
        self.prediction_stream = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 156) 
        )

    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)
            
        features = self.input_layer(x)
        features = self.res_blocks(features)
        
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        
        q_vals = value + (advantage - advantage.mean(dim=1, keepdim=True))
        
        # Return both Q-Values (for acting) and Predictions (for learning)
        # We only use predictions during training
        predictions = self.prediction_stream(features)
        
        return q_vals, predictions

def get_batch_input(hands_list, played_cards_list, table_bounds_list, pass_flags_list, hand_sizes_list, device):
    # ... (Keep your existing implementation here) ...
    # Ensure this matches your provided file exactly
    batch_size = len(hands_list)
    hand_tensor = torch.zeros((batch_size, 52), device=device)
    batch_indices = []
    card_indices = []
    for i, hand in enumerate(hands_list):
        for card in hand:
            batch_indices.append(i)
            card_indices.append(card)
    if batch_indices:
        hand_tensor[batch_indices, card_indices] = 1.0
        
    played_tensor = torch.tensor(played_cards_list, dtype=torch.float32, device=device)
    bounds_tensor = torch.full((batch_size, 8), -1.0, dtype=torch.float32, device=device)
    pass_tensor = torch.tensor(pass_flags_list, dtype=torch.float32, device=device)
    size_tensor = torch.tensor(hand_sizes_list, dtype=torch.float32, device=device) / 13.0
    for i, bounds in enumerate(table_bounds_list):
        for suit in range(4):
            low, high = bounds[suit]
            if low is not None:
                bounds_tensor[i, suit*2] = (low - 6.0) / 6.0
            if high is not None:
                bounds_tensor[i, suit*2+1] = (high - 6.0) / 6.0
                
    return torch.cat([hand_tensor, played_tensor, bounds_tensor, pass_tensor, size_tensor], dim=1)