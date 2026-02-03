# app/utils.py
from utils.constant import RANKS, SUITS

def card_parts(card_id):
    suit_idx, rank_idx = divmod(card_id, 13)
    return RANKS[rank_idx], SUITS[suit_idx]

def to_bot_card(card_id):
    if card_id is None: return None
    suit_idx, rank_idx = divmod(card_id, 13)
    return (suit_idx, rank_idx + 1)

def from_bot_card(action_idx):
    """Converts Bot Action to UI ID (0-51)."""
    if action_idx is None: return None
    
    # 1. Handle "Pass" (Action 52)
    if isinstance(action_idx, int) and action_idx == 52:
        return None
        
    # 2. Handle Tuple Return (Suit Int, Rank Int)
    if isinstance(action_idx, tuple):
        suit_idx, rank_val = action_idx
        return suit_idx * 13 + (rank_val - 1)
        
    # 3. Handle Raw Index (0-51)
    if isinstance(action_idx, int):
        if action_idx >= 52: return None
        return action_idx
        
    return None
