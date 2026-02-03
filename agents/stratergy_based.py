class LooseAggressive:
    """
    Description: Raises aggressively with a wide range of hands[cite: 573].
    BadaamSath Logic: Prioritizes opening suits (7s) or playing extreme cards.
    """
    def select_action(self, state):
        legal_moves = state['legal_moves']
        
        def aggression_score(card):
            # Check for NoneType (Pass action)
            if card is None:
                return -1
            
            rank = card[0] 
            if rank == 7: return 10
            if rank in [1, 13]: return 8
            return abs(rank - 7)

        scored_moves = [(aggression_score(m), m) for m in legal_moves]
        scored_moves.sort(key=lambda x: x[0], reverse=True)
        return scored_moves[0][1]

class TightPassive:
    """
    Description: Calls with good hands, folds with most, rarely raises[cite: 573].
    BadaamSath Logic: Hoards cards; keeps sequences tight near the 7s.
    """
    def select_action(self, state):
        legal_moves = state['legal_moves']
        
        def passivity_score(card):
            if card is None:
                return 20 # High priority to Passing if it's an option to hoard
                
            rank = card[0]
            if rank == 7: return 0
            return 14 - abs(rank - 7)

        scored_moves = [(passivity_score(m), m) for m in legal_moves]
        scored_moves.sort(key=lambda x: x[0], reverse=True)
        return scored_moves[0][1]

class CandidStatistician:
    """
    Description: Raises with good hands, folds with weak[cite: 573].
    BadaamSath Logic: Plays from the suit it holds the most cards in.
    """
    def select_action(self, state):
        legal_moves = state['legal_moves']
        hand = state['hand']
        
        # Count suit distribution safely
        suit_counts = {}
        for card in hand:
            if card is not None:
                suit = card[1]
                suit_counts[suit] = suit_counts.get(suit, 0) + 1
            
        scored_moves = []
        for move in legal_moves:
            if move is None:
                score = -1 
            else:
                score = suit_counts.get(move[1], 0)
            scored_moves.append((score, move))
            
        scored_moves.sort(key=lambda x: x[0], reverse=True)
        return scored_moves[0][1]