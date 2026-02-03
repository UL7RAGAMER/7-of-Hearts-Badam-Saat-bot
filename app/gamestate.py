from utils.helper_functions import to_bot_card


class GameState:
    def __init__(self):
        self.reset()

    def reset(self):
        self.my_hand = set()
        self.played_cards = set()
        self.table_bounds = {i: [None, None] for i in range(4)} 
        self.history = []
        self.current_turn = 0 

    def play_card(self, cid, user=False):
        suit_idx, rank_idx = divmod(cid, 13) 
        rank_val = rank_idx + 1              
        
        prev_bounds = list(self.table_bounds[suit_idx])
        
        self.history.append({
            "cid": cid,
            "user": user,
            "suit": suit_idx,
            "prev_bounds": prev_bounds
        })

        low, high = self.table_bounds[suit_idx]
        if low is None:
            self.table_bounds[suit_idx] = [rank_val, rank_val]
        else:
            self.table_bounds[suit_idx][0] = min(low, rank_val)
            self.table_bounds[suit_idx][1] = max(high, rank_val)

        self.played_cards.add(cid)
        if user and cid in self.my_hand:
            self.my_hand.remove(cid)

        self.current_turn = (self.current_turn + 1) % 4

    def undo_last_move(self):
        if not self.history: return
        last_move = self.history.pop()
        
        if last_move['cid'] in self.played_cards:
            self.played_cards.remove(last_move['cid'])
            
        self.table_bounds[last_move['suit']] = last_move['prev_bounds']
        
        if last_move['user']:
            self.my_hand.add(last_move['cid'])
            
        self.current_turn = (self.current_turn - 1) % 4

    def legal_moves(self, hand):
        moves = []
        for cid in hand:
            suit_idx, rank_idx = divmod(cid, 13)
            rank_val = rank_idx + 1 
            
            low, high = self.table_bounds[suit_idx]
            
            if rank_val == 7:
                moves.append(cid)
            elif low is not None and (rank_val == low - 1 or rank_val == high + 1):
                moves.append(cid)
        return moves

    def get_bot_state(self):
        bot_hand = [to_bot_card(c) for c in self.my_hand]
        
        bot_board = []
        for s_idx in range(4):
            low, high = self.table_bounds[s_idx]
            bot_board.append({
                'min': low, 
                'max': high
            })

        ui_legal = self.legal_moves(self.my_hand)
        bot_legal = [to_bot_card(c) for c in ui_legal]
        if not bot_legal:
            bot_legal = [None] 

        total_played = len(self.played_cards)
        my_hand_size = len(self.my_hand)
        remaining = 52 - total_played - my_hand_size
        avg_opp = remaining // 3
        
        return {
            'hand': bot_hand,
            'board': bot_board,
            'legal_moves': bot_legal,
            'player_hand_counts': [my_hand_size, avg_opp, avg_opp, avg_opp], 
            'turn': 0, 
            'history': [] 
        }
