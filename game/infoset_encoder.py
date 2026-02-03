import torch
import numpy as np

class BadaamSathEncoder:
    def __init__(self):
        # 52 cards + turn info + opponent card counts
        # Hand (52) + Board (52) + Player ID (4) + Opponent Counts (4)
        self.input_dim = 52 + 52 + 4 + 4 
        self.action_dim = 53 # Each card in a deck is a potential action

    def encode(self, state, player_id):
        """
        Converts game engine state into a flat tensor.
        """
        # 1. Encode Hand (Multi-hot: 1 if card is in hand, 0 otherwise)
        hand_vec = np.zeros(52)
        for suit, rank in state['hand']:
            idx = suit * 13 + (rank - 1)
            hand_vec[idx] = 1

        # 2. Encode Board (Multi-hot: 1 if card is on board, 0 otherwise)
        board_vec = np.zeros(52)
        for suit in range(4):
            b_min = state['board'][suit]['min']
            b_max = state['board'][suit]['max']
            if b_min is not None:
                for r in range(b_min, b_max + 1):
                    idx = suit * 13 + (r - 1)
                    board_vec[idx] = 1

        # 3. Encode Current Player ID (One-hot)
        player_vec = np.zeros(4)
        player_vec[player_id] = 1

        hand_sizes = []
        for p in range(4):
            # We get this count from the engine
            count = state['player_hand_counts'][p]
            hand_sizes.append(count / 13.0) 
        
        count_vec = np.array(hand_sizes)

        # Combine all features into one flat vector
        combined = np.concatenate([hand_vec, board_vec, player_vec, count_vec])
        return torch.FloatTensor(combined)

    def action_to_idx(self, action):
        """Converts (suit, rank) to a single integer 0-51."""
        if action is None: return 52 # Reserved for 'Pass'
        suit, rank = action
        return suit * 13 + (rank - 1)

    def idx_to_action(self, idx):
        """Converts integer 0-51 back to (suit, rank)."""
        if idx == 52: return None
        suit = idx // 13
        rank = (idx % 13) + 1
        return (suit, rank)