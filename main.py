import torch
import argparse
import sys
import os
import random
from simulation.trainer import train
from simulation.engine import BadaamSathEngine
from agent.dqn import DQNAgent
from agent.model import get_batch_input

# UI Helpers
SUITS = ["Hearts", "Diamonds", "Clubs", "Spades"]
RANKS = ["A", "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K"]

def get_card_name(card_id): 
    if card_id is None or card_id == 52: return "Pass"
    suit = SUITS[card_id // 13]
    rank = RANKS[card_id % 13]
    return f"{rank} of {suit}"

def play_mode():
    engine = BadaamSathEngine() # 
    # 1. Point to the v4 weights
    weights_path = "agent/weights/badaamsath_dueling_v4.pth"
    
    # 2. Initialize Bots with 119-dim state space
    # We create bots for Player 1, 2, and 3
    bots = {i: DQNAgent(state_dim=119, action_dim=53) for i in range(1, 4)} # [cite: 4]
    
    print("\n" + "="*50)
    print("      BADAAM SATH: CHALLENGE MODE (v4)")
    print("="*50)
    print(" PLAYER 0: YOU (The Human)")
    print(" PLAYER 1: SMART BOT (Competitive Agent)")
    print(" PLAYER 2: RANDOM BOT (Chaos)")
    print(" PLAYER 3: RANDOM BOT (Chaos)")
    print("-" * 50)

    # 3. Load v4 weights for the Smart Bot
    if os.path.exists(weights_path):
        print(f"[*] Loading Smart Brain for Player 1...")
        bots[1].load_weights(weights_path) # [cite: 4]
    else:
        print(f"[!] Warning: {weights_path} not found. Smart Bot will be random.")

    engine.reset_game() # 
    starter = engine.current_turn
    print(f"\n[*] Player {starter} holds the 7 of Hearts and starts.")

    while not engine.game_over:
        curr = engine.current_turn
        legal = engine.get_legal_moves(curr) # 
        
        # --- HUMAN TURN (Player 0) ---
        if curr == 0:
            print(f"\n--- YOUR TURN (Player 0) ---")
            hand = sorted(engine.hands[0]) # 
            
            # Show the board status for suits
            print(f"Table Bounds: {engine.table_bounds}")
            print(f"Your Hand: {[(c, get_card_name(c)) for c in hand]}")
            
            if not legal:
                print(">>> No legal moves. You must pass.")
                move = None
                input("Press Enter to continue...")
            else:
                readable_legal = [(m, get_card_name(m)) for m in legal]
                print(f"LEGAL MOVES: {readable_legal}")
                
                valid_input = False
                while not valid_input:
                    user_in = input("Enter Card ID to play: ").strip()
                    if user_in.isdigit() and int(user_in) in legal:
                        move = int(user_in)
                        valid_input = True
                    else:
                        print("Invalid selection. Please choose an ID from the legal moves list.")
            
            engine.apply_move(0, move) # 

        # --- BOT TURNS (Players 1, 2, 3) ---
        else:
            # 1. Get Pass Flags (Full list of 4 as per v4 training)
            pass_flags = engine.last_action_was_pass.copy() # 

            # 2. Get Hand Sizes of the 3 Opponents relative to current bot
            hand_sizes = [len(engine.hands[(curr + j) % 4]) for j in range(1, 4)] # 

            # 3. Format v4 State Input (5 data arguments)
            state_tensor = get_batch_input(
                [engine.hands[curr]], 
                [engine.played_cards], 
                [engine.table_bounds], 
                [pass_flags], 
                [hand_sizes], # Added hand sizes for v4
                bots[curr].device
            ) # [cite: 5]
            
            # 4. Strategy Selection
            if curr == 1:
                # Use trained brain (v4)
                action = bots[curr].select_action(state_tensor, legal, epsilon=0.0) # [cite: 4]
                p_type = "Smart Bot"
            else:
                # Play randomly to mimic casual opponents
                action = random.choice(legal) if legal else 52
                p_type = "Random Bot"

            move = action if action != 52 else None
            print(f"Player {curr} ({p_type}) plays: {get_card_name(move)}")
            engine.apply_move(curr, move) # 

    # --- RESULTS ---
    print(f"\n" + "*"*40)
    print(f"GAME OVER! WINNER: PLAYER {engine.winner}")
    print("*"*40)
    
    for s in engine.get_scores(): # 
        role = {0: "YOU", 1: "SMART BOT", 2: "RANDOM", 3: "RANDOM"}[s['player']]
        win_label = " (WINNER)" if s['player'] == engine.winner else ""
        print(f"Player {s['player']}{win_label:<9} | {role:<10} | Cards: {s['card_count']:<2} | Score: {s['points']} pts")
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "play"], default="train")
    args = parser.parse_args()

    if args.mode == "train":
        train()
    elif args.mode == "play":
        play_mode()

if __name__ == "__main__":
    main()