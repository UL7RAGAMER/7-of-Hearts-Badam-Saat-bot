import numpy as np

# Suit and Rank Constants
HEARTS, DIAMONDS, CLUBS, SPADES = 0, 1, 2, 3
SUITS = [HEARTS, DIAMONDS, CLUBS, SPADES]
RANKS = list(range(1, 14))  # 1 (Ace) to 13 (King)

class BadaamSathEngine:
    def __init__(self, num_players=4):
        self.num_players = num_players
        self.reset()

    def reset(self):
        """Initializes a new game, deals cards, and finds the starter."""
        # 1. Create and shuffle 52-card deck
        deck = [(s, r) for s in SUITS for r in RANKS]
        np.random.shuffle(deck)
        
        # 2. Deal 13 cards to each of the 4 players
        self.hands = [set(deck[i::self.num_players]) for i in range(self.num_players)]
        
        # 3. Board state: tracks the min and max rank played for each suit
        self.board = {s: {"min": None, "max": None} for s in SUITS}
        
        # 4. Turn management: 7 of Hearts starts the game
        self.current_player = next(i for i, h in enumerate(self.hands) if (HEARTS, 7) in h)
        self.game_started = False
        self.done = False
        
        # 5. CRITICAL FIX: Track consecutive passes to detect game end
        self.consecutive_passes = 0
        
        return self.get_state(self.current_player)

    def get_legal_moves(self, player_id):
        """Identifies playable cards or 'Pass'."""
        hand = self.hands[player_id]
        legal_moves = []
        
        if not self.game_started:
            # The 7 of Hearts MUST be the first card played
            return [(HEARTS, 7)] if (HEARTS, 7) in hand else []

        for suit, rank in hand:
            if rank == 7:
                # Any 7 can start its suit sequence once 7H is played
                if self.board[suit]["min"] is None:
                    legal_moves.append((suit, rank))
            else:
                # Can play if rank is adjacent to current suit sequence on board
                b_min, b_max = self.board[suit]["min"], self.board[suit]["max"]
                if b_min is not None and (rank == b_min - 1 or rank == b_max + 1):
                    legal_moves.append((suit, rank))

        # If no legal card plays exist, the only option is to 'Pass' (represented as None)
        return legal_moves if legal_moves else [None]

    def apply_move(self, action):
        """Updates the game state based on the chosen action."""
        p_id = self.current_player
        
        if action is not None:
            # A card was played, reset consecutive passes
            self.consecutive_passes = 0
            
            suit, rank = action
            # Update suit sequence boundaries
            if rank == 7:
                self.board[suit]["min"] = self.board[suit]["max"] = 7
                if suit == HEARTS: 
                    self.game_started = True
            # CRITICAL FIX: Proper None checking before comparison
            elif self.board[suit]["min"] is not None:
                if rank < self.board[suit]["min"]:
                    self.board[suit]["min"] = rank
                elif rank > self.board[suit]["max"]:
                    self.board[suit]["max"] = rank
            # This shouldn't happen (only 7s can start a suit)
            else:
                # Safety fallback - treat as starting the suit
                self.board[suit]["min"] = self.board[suit]["max"] = rank
            
            # Remove card from hand
            self.hands[p_id].remove(action)
            
            # Check win condition: player emptied their hand
            if not self.hands[p_id]:
                self.done = True
                return self.done
        else:
            # Player passed
            self.consecutive_passes += 1
            
            # CRITICAL FIX: If all players pass consecutively, game is stuck
            # This happens when no one can play but everyone still has cards
            # In real Sevens, this means the game ends
            if self.consecutive_passes >= self.num_players:
                self.done = True
                return self.done
        
        # Move to next player clockwise
        self.current_player = (self.current_player + 1) % self.num_players
        return self.done

    def get_utility(self, player_id):
        """Returns terminal payoff based on remaining card values."""
        if not self.done: 
            return 0
        
        # Standard scoring: negative points for remaining cards
        # Ace=1, 2-10=face value, J=11, Q=12, K=13
        points = sum(rank for _, rank in self.hands[player_id])
        return -points

    def get_state(self, player_id):
        return {
            "hand": list(self.hands[player_id]),
            "board": self.board,
            "turn": self.current_player,
            "player_hand_counts": [len(h) for h in self.hands],
            "legal_moves": self.get_legal_moves(player_id)
        }