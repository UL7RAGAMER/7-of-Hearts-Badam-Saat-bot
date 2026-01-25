import torch
import numpy as np
import time
import random
from simulation.engine import BadaamSathEngine
from agent.dqn import DQNAgent
from agent.model import get_batch_input

# --- Configuration ---
MODEL_PATH = "agent/weights/badaamsath_dueling_v4.pth"
NUM_ENVS = 1000  
TOTAL_GAMES_TARGET = 10000

def evaluate_parallel():
    print(f"[*] Initializing {TOTAL_GAMES_TARGET} games across {NUM_ENVS} environments...")
    agent = DQNAgent(state_dim=119, action_dim=53, device="cuda")
    
    try:
        agent.load_weights(MODEL_PATH)
        agent.policy_net.eval()
    except Exception as e:
        print(f"[!] Error loading weights: {e}")
        return

    envs = [BadaamSathEngine() for _ in range(NUM_ENVS)]
    games_finished = 0
    total_wins = 0
    total_illegal_moves = 0  # <--- THE NEW COUNTER
    
    print(f"[*] Evaluating on RTX 5070 Ti...")
    start_time = time.time()

    while games_finished < TOTAL_GAMES_TARGET:
        active_envs = [i for i, env in enumerate(envs) if not env.game_over]
        
        if not active_envs:
            for env in envs:
                if env.winner == 0: total_wins += 1
                games_finished += 1
                env.reset_game()
            continue

        p0_indices = []
        p0_h, p0_p, p0_b, p0_pf, p0_hs, p0_leg = [], [], [], [], [], []

        for idx in active_envs:
            env = envs[idx]
            curr = env.current_turn
            legal = env.get_legal_moves(curr) #

            if curr == 0:
                p0_indices.append(idx)
                p0_h.append(env.hands[0])
                p0_p.append(env.played_cards)
                p0_b.append(env.table_bounds)
                p0_pf.append(env.last_action_was_pass.copy())
                p0_hs.append([len(env.hands[j]) for j in range(1, 4)])
                p0_leg.append(legal)
            else:
                move = random.choice(legal) if legal else None
                env.apply_move(curr, move)

        if p0_indices:
            state_batch = get_batch_input(p0_h, p0_p, p0_b, p0_pf, p0_hs, agent.device) #
            actions = agent.select_actions_batch(state_batch, p0_leg, epsilon=0.0) #
            
            for i, (idx, action) in enumerate(zip(p0_indices, actions)):
                # --- ILLEGAL MOVE CHECK ---
                legal_moves_for_this_env = p0_leg[i]
                
                # If the agent picks a card it shouldn't, or tries to pass when it has moves
                if action not in legal_moves_for_this_env and not (action == 52 and not legal_moves_for_this_env):
                    total_illegal_moves += 1
                
                move = action if action != 52 else None
                envs[idx].apply_move(0, move)

        if games_finished > 0 and games_finished % 1000 == 0:
            current_wr = (total_wins / games_finished) * 100
            print(f"Finished {games_finished}/{TOTAL_GAMES_TARGET} | Win Rate: {current_wr:.1f}% | Illegal Moves: {total_illegal_moves}")

    end_time = time.time()
    final_wr = (total_wins / games_finished) * 100
    print(f"\n[!] FINAL RESULTS")
    print(f"Total Games: {games_finished} | Win Rate: {final_wr:.1f}%")
    print(f"Total Illegal Moves Detected: {total_illegal_moves}") # If this is > 0, masking is broken
    print(f"Evaluation Speed: {(games_finished/(end_time - start_time)):.1f} games/sec")

if __name__ == "__main__":
    evaluate_parallel()