import torch
import numpy as np
import os
import random
from simulation.engine import BadaamSathEngine
from agent.dqn import DQNAgent
from agent.model import get_batch_input
from data.replay_buffer import PrioritizedReplayBuffer

# --- Updated Hyperparameters ---
# Note: state_dim increased to 122 to accommodate hand sizes and pass memory
STATE_DIM = 119 
NUM_ENVS = 2048
MAX_EPISODES = 40000 # Increased for more complex strategy learning
BATCH_SIZE = 4096
GAMMA = 0.99
EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY = 0.9999 # Slower decay for more exploration in self-play
TARGET_UPDATE = 100
SNAPSHOT_INTERVAL = 1000 
WEIGHTS_PATH = "agent/weights/badaamsath_dueling_v4.pth"
LEARNING_RATE = 1e-4
UPDATE_EVERY_N_STEPS = 16
BETA_START = 0.4
BETA_FRAMES = 100000

torch.set_float32_matmul_precision('high')

def get_opponent_ground_truth(env, player_idx):
    opp_vec = torch.zeros(156)
    opp_indices = [(player_idx + i) % 4 for i in range(1, 4)]
    for i, opp_id in enumerate(opp_indices):
        hand = env.hands[opp_id]
        if hand:
            offset = i * 52
            indices = [c + offset for c in hand]
            opp_vec[indices] = 1.0
    return opp_vec

def train():
    os.makedirs("agent/weights/snapshots", exist_ok=True)
    
    # Updated state_dim to 122
    agent = DQNAgent(state_dim=STATE_DIM, action_dim=53, gamma=GAMMA, lr=LEARNING_RATE)
    memory = PrioritizedReplayBuffer(capacity=200000)
    
    envs = [BadaamSathEngine() for _ in range(NUM_ENVS)]
    
    epsilon = EPS_START
    total_episodes = 0
    total_steps = 0
    steps_since_train = 0
    
    win_count = 0
    batch_penalty_points = 0  
    episodes_in_current_batch = 0
    steps_at_last_log = 0     

    print(f"[*] Starting 'Competitive Self-Play' Training ({NUM_ENVS} envs)...")

    while total_episodes < MAX_EPISODES:
        beta = min(1.0, BETA_START + total_steps * (1.0 - BETA_START) / BETA_FRAMES)

        # --- STEP 1: COLLECT DATA ---
        # Added 'hs' for hand sizes to 'collect'
        collect = {i: {'h':[], 'p':[], 'b':[], 'pf':[], 'hs':[], 'idx':[], 'leg':[]} for i in range(4)}

        for i, env in enumerate(envs):
            if env.game_over:
                if env.winner == 0: win_count += 1
                p0_points = sum((c % 13) + 1 for c in env.hands[0])
                batch_penalty_points += p0_points
                
                total_episodes += 1
                episodes_in_current_batch += 1
                env.reset_game()
                
                if total_episodes % 100 == 0:
                    avg_win = (win_count / episodes_in_current_batch) * 100
                    avg_pts = batch_penalty_points / episodes_in_current_batch
                    steps_delta = total_steps - steps_at_last_log
                    avg_speed = steps_delta / episodes_in_current_batch
                    print(f"Ep {total_episodes} | Win: {avg_win:.1f}% | AvgPts: {avg_pts:.1f} | Speed: {avg_speed:.1f} mv/gm | Eps: {epsilon:.2f}")
                    win_count, batch_penalty_points, episodes_in_current_batch = 0, 0, 0
                    steps_at_last_log = total_steps
                    agent.save_snapshot(total_episodes)
                
                epsilon = max(EPS_END, epsilon * EPS_DECAY)

            curr = env.current_turn
            legal = env.get_legal_moves(curr)
            
            # Opponent Hand Sizes (Normalized by current player)
            hs = [len(env.hands[(curr + j) % 4]) for j in range(1, 4)]

            collect[curr]['h'].append(env.hands[curr])
            collect[curr]['p'].append(env.played_cards)
            collect[curr]['b'].append(env.table_bounds)
            collect[curr]['pf'].append(env.last_action_was_pass.copy())
            collect[curr]['hs'].append(hs)
            collect[curr]['idx'].append(i)
            collect[curr]['leg'].append(legal)

        # --- STEP 2: COMPETITIVE BATCH INFERENCE ---
        actions_to_apply = {} 
        p0_state_batch = None
        p0_indices_map = {} 

        for pid in range(4):
            data = collect[pid]
            if not data['idx']: continue
            
            # Pass all 5 state components to get_batch_input
            batch_tensor = get_batch_input(data['h'], data['p'], data['b'], data['pf'], data['hs'], agent.device)
            
            if pid == 0:
                p0_state_batch = batch_tensor 
                p0_indices_map = {env_idx: row_idx for row_idx, env_idx in enumerate(data['idx'])}
                actions = agent.select_actions_batch(batch_tensor, data['leg'], epsilon)
            else:
                # SELF-PLAY: Opponents play with their own logic but a fixed competitive epsilon
                # This stops them from being random and starts testing the agent's defense
                actions = agent.select_actions_batch(batch_tensor, data['leg'], epsilon=0.15)
                
            for idx, action in zip(data['idx'], actions):
                actions_to_apply[idx] = action

        # --- STEP 3: APPLY & STRATEGIC REWARD ---
        next_s_data = {'h': [], 'p': [], 'b': [], 'pf': [], 'hs': [], 'metadata': []}
        
        for i, env in enumerate(envs):
            if i not in actions_to_apply: continue
            
            action = actions_to_apply[i]
            move = action if action != 52 else None
            
            if env.current_turn == 0 and i in p0_indices_map:
                row_idx = p0_indices_map[i]
                s_t = p0_state_batch[row_idx].cpu()
                opp_gt = get_opponent_ground_truth(env, 0)

                reward = 0.0
                if env.first_move:
                    reward += 5.0 if move == env.SEVEN_OF_HEARTS else -10.0

                if move is not None:
                    rank, suit = move % 13, move // 13
                    reward += 0.5 # Small survival reward
                    
                    # 1. SCORE REDUCTION
                    if rank >= 10: reward += 5.0 # Priority dump for J, Q, K
                    
                    # 2. THE SQUEEZE (Forcing Pass)
                    next_player = (env.current_turn + 1) % 4
                    if not env.get_legal_moves(next_player):
                        reward += 20.0 # High reward for making others pass
                    
                    # 3. COMPETITIVE BLOCKING
                    if rank == 5 or rank == 7:
                        opp_sizes = [len(env.hands[j]) for j in range(1, 4)]
                        danger_zone = min(opp_sizes) < 4 # Someone is about to win
                        
                        follower = (rank - 1) if rank == 5 else (rank + 1)
                        has_follower = (suit * 13 + follower) in env.hands[0]
                        
                        if not has_follower and danger_zone:
                            # If playing this helps someone win and doesn't help me...
                            reward -= 15.0 # Stop cleaning opponent hands!
                        elif has_follower:
                            reward += 4.0 # Good to play if it unlocks your own cards
                else:
                    reward -= 10.0 # Heavier Pass Penalty (Force efficient card usage)
                
                env.apply_move(0, move)

                if env.game_over:
                    reward += 100.0 if env.winner == 0 else -(sum((c % 13) + 1 for c in env.hands[0]))
                
                # Prep next state data
                next_hs = [len(env.hands[j]) for j in range(1, 4)]
                next_s_data['h'].append(env.hands[0])
                next_s_data['p'].append(env.played_cards)
                next_s_data['b'].append(env.table_bounds)
                next_s_data['pf'].append(env.last_action_was_pass.copy())
                next_s_data['hs'].append(next_hs)
                next_s_data['metadata'].append((s_t, action, reward, env.game_over, opp_gt))
                
                steps_since_train += 1
                total_steps += 1
            else:
                env.apply_move(env.current_turn, move)

        # --- STEP 4: BUFFER & TRAIN ---
        if next_s_data['h']:
            # Ensure get_batch_input here also has all 5 arguments
            next_s_batch = get_batch_input(
                next_s_data['h'], next_s_data['p'], next_s_data['b'], 
                next_s_data['pf'], next_s_data['hs'], agent.device
            ).cpu()
            
            for k in range(len(next_s_data['h'])):
                s_t, action, reward, done, opp_gt = next_s_data['metadata'][k]
                memory.push(s_t, action, reward, next_s_batch[k], done, opp_gt)

        if steps_since_train >= UPDATE_EVERY_N_STEPS:
            agent.update_model(BATCH_SIZE, memory, beta=beta)
            steps_since_train = 0
            if total_steps % (TARGET_UPDATE * UPDATE_EVERY_N_STEPS) == 0:
                agent.update_target_network()