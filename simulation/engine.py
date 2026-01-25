import random

class BadaamSathEngine:
    def __init__(self):
        # Mapping: 0-12 Hearts, 13-25 Diamonds, 26-38 Clubs, 39-51 Spades
        # 7 of Hearts is ID 6 (Suit 0, Rank 6)
        self.suits = ['Hearts', 'Diamonds', 'Clubs', 'Spades']
        self.ranks = ['A', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K']
        self.SEVEN_OF_HEARTS = 6 
        self.reset_game()

    def get_scores(self):
            """Calculates penalty points for each player based on remaining cards."""
            scores = []
            for i, hand in enumerate(self.hands):
                # Sum of (rank_idx + 1) for each card in hand
                # Rank index 0 (Ace) = 1 point, index 12 (King) = 13 points
                points = sum((card_id % 13) + 1 for card_id in hand)
                scores.append({
                    "player": i,
                    "points": points,
                    "card_count": len(hand)
                })
            return scores
    
    def reset_game(self):
        """Initializes a new game state following Badaam Sath rules."""
        self.deck = list(range(52))
        random.shuffle(self.deck)
        
        # Deal 13 cards to 4 players
        self.hands = [sorted(self.deck[i*13:(i+1)*13]) for i in range(4)]
        
        # Table state: [min_rank, max_rank] per suit
        self.table_bounds = {suit_idx: [None, None] for suit_idx in range(4)}
        self.played_cards = [0] * 52
        
        # Rule: Player with 7 of Hearts starts
        self.current_turn = self._find_starting_player()
        self.first_move = True
        self.game_over = False
        self.winner = None
        self.last_action_was_pass = [0] * 4
    def _find_starting_player(self):
        for i, hand in enumerate(self.hands):
            if self.SEVEN_OF_HEARTS in hand:
                return i
        return 0

    def get_legal_moves(self, player_idx):
        """Returns valid moves. Enforces the 7 of Hearts as the mandatory opener."""
        hand = self.hands[player_idx]
        
        # Mandatory Start Rule
        if self.first_move:
            if self.SEVEN_OF_HEARTS in hand:
                return [self.SEVEN_OF_HEARTS]
            return [] # Should not happen if current_turn is correct

        legal_moves = []
        for card_id in hand:
            suit_idx = card_id // 13
            rank_idx = card_id % 13
            low, high = self.table_bounds[suit_idx]

            # Rule 1: Any 7 can start its suit
            if rank_idx == 6:
                legal_moves.append(card_id)
            # Rule 2: If suit started, play adjacent rank
            elif low is not None:
                if rank_idx == low - 1 or rank_idx == high + 1:
                    legal_moves.append(card_id)

        return legal_moves

    def apply_move(self, player_idx, card_id):
        """Processes move and updates first_move status."""

        # --- FIX: Handle Validation Correctly ---
        legal_moves = self.get_legal_moves(player_idx)
        
        # 1. If playing a card, verify it is legal
        if card_id is not None and card_id not in legal_moves:
            return False
            
        # 2. (Optional Strict Rule) If passing, ensure they truly have no moves
        # If you want to enforce "Must Play", uncomment the next two lines:
        # if card_id is None and len(legal_moves) > 0:
        #     return False

        # --- PASS LOGIC ---
        if card_id is None: # Pass
            self.last_action_was_pass[player_idx] = 1 # MARK AS PASSED
            self.current_turn = (self.current_turn + 1) % 4
            return True

        # --- PLAY LOGIC ---
        # If they played a card, reset their pass flag (they are back in the game)
        self.last_action_was_pass[player_idx] = 0
        
        # Update Board
        suit_idx = card_id // 13
        rank_idx = card_id % 13
        
        if self.table_bounds[suit_idx][0] is None:
            self.table_bounds[suit_idx] = [rank_idx, rank_idx]
        else:
            self.table_bounds[suit_idx][0] = min(self.table_bounds[suit_idx][0], rank_idx)
            self.table_bounds[suit_idx][1] = max(self.table_bounds[suit_idx][1], rank_idx)

        # Update Player State
        self.hands[player_idx].remove(card_id)
        self.played_cards[card_id] = 1
        self.first_move = False 

        if not self.hands[player_idx]:
            self.game_over = True
            self.winner = player_idx

        self.current_turn = (self.current_turn + 1) % 4
        return True
    
    def get_state_for_agent(self, player_idx):
        return {
            "own_hand": self.hands[player_idx],
            "table_bounds": self.table_bounds,
            "played_cards": self.played_cards,
            "others_hand_sizes": [len(self.hands[i]) for i in range(4) if i != player_idx]
        }