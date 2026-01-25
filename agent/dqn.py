import os
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
from agent.model import BadaamSathDuelingDQN

class DQNAgent:
    def __init__(self, state_dim=115, action_dim=53, lr=1e-4, gamma=0.99, device=None):
        if device:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
        self.gamma = gamma
        self.action_dim = action_dim
        
        self.policy_net = BadaamSathDuelingDQN(state_dim, action_dim).to(self.device)
        self.target_net = BadaamSathDuelingDQN(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.scaler = torch.amp.GradScaler('cuda') if self.device.type == 'cuda' else None

    def select_action(self, state_tensor, legal_moves, epsilon):
        """Single-instance inference"""
        if random.random() < epsilon:
            return random.choice(legal_moves) if legal_moves else 52
        
        self.policy_net.eval() 
        with torch.no_grad():
            state_tensor = state_tensor.to(self.device)
            
            # --- FIX: ROBUST DIMENSION HANDLING ---
            # Only add batch dimension if it's missing (1D -> 2D)
            # If it is already (1, 115), do NOT unsqueeze again.
            if state_tensor.dim() == 1:
                state_tensor = state_tensor.unsqueeze(0)
            
            if self.device.type == 'cuda':
                with torch.amp.autocast('cuda'):
                    # Tuple unpacking: we only need q_values [0]
                    q_values = self.policy_net(state_tensor)[0]
            else:
                q_values = self.policy_net(state_tensor)[0]
            
            # Since q_values will be (1, 53), we take [0] to get 1D
            q_values = q_values[0]
            
            mask = torch.full((self.action_dim,), float('-inf'), device=self.device)
            if not legal_moves:
                mask[52] = 0
            else:
                for move in legal_moves:
                    mask[move] = 0
            
            masked_q_values = q_values + mask
            action = masked_q_values.argmax().item()
        
        self.policy_net.train() 
        return action

    @torch.no_grad()
    def select_actions_batch(self, states, legal_moves_list, epsilon):
        """
        Optimized batch inference with Vectorized Masking.
        """
        if isinstance(states, list):
            if not states: return []
            device_states = torch.stack(states).to(self.device, non_blocking=True)
        else:
            device_states = states

        num_ready = device_states.shape[0]
        
        self.policy_net.eval()
        
        # 1. Inference
        if self.device.type == 'cuda':
            with torch.amp.autocast('cuda'):
                # Tuple unpacking for V2 architecture: (Q, Preds)
                q_values = self.policy_net(device_states)[0]
        else:
            q_values = self.policy_net(device_states)[0]
            
        # 2. Vectorized Masking
        mask = torch.full((num_ready, self.action_dim), float('-inf'), device=self.device)
        
        rows = []
        cols = []
        
        for i, moves in enumerate(legal_moves_list):
            if not moves:
                rows.append(i)
                cols.append(52) # Pass
            else:
                rows.extend([i] * len(moves))
                cols.extend(moves)
        
        if rows:
            row_t = torch.tensor(rows, device=self.device, dtype=torch.long)
            col_t = torch.tensor(cols, device=self.device, dtype=torch.long)
            mask[row_t, col_t] = 0.0
            
        masked_q = q_values + mask
        greedy_actions = masked_q.argmax(dim=1).cpu().numpy()
        
        # 3. Epsilon-Greedy
        final_actions = []
        for i in range(num_ready):
            if random.random() < epsilon:
                final_actions.append(random.choice(legal_moves_list[i]) if legal_moves_list[i] else 52)
            else:
                final_actions.append(greedy_actions[i])
                
        self.policy_net.train()
        return final_actions

    def update_model(self, batch_size, memory, beta=0.4):
        if len(memory) < batch_size: return None
        
        self.policy_net.train()
        
        batch = memory.sample(batch_size, beta)
        states, actions, rewards, next_states, dones, opp_hands, indices, weights = batch

        states = torch.stack(states).to(self.device, non_blocking=(self.device.type=='cuda'))
        actions = torch.tensor(actions).to(self.device, non_blocking=(self.device.type=='cuda'))
        rewards = torch.tensor(rewards).to(self.device, non_blocking=(self.device.type=='cuda'))
        next_states = torch.stack(next_states).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).to(self.device)
        opp_hands = torch.stack(opp_hands).to(self.device).float()
        weights = weights.to(self.device)

        if self.device.type == 'cuda':
            with torch.amp.autocast('cuda'):
                # Forward Pass (get Q and Predictions)
                current_q, pred_logits = self.policy_net(states)
                current_q = current_q.gather(1, actions.unsqueeze(1)).squeeze(1)
                
                with torch.no_grad():
                    # Target Q (only needs Q-head)
                    next_q_vals = self.target_net(next_states)[0]
                    next_actions = self.policy_net(next_states)[0].argmax(dim=1, keepdim=True)
                    next_q = next_q_vals.gather(1, next_actions).squeeze(1)
                    target_q = rewards + (1 - dones) * self.gamma * next_q
                
                # DQN Loss
                td_errors = torch.abs(current_q - target_q).detach()
                dqn_loss = (weights * F.smooth_l1_loss(current_q, target_q, reduction='none')).mean()
                
                # Aux Loss (Prediction)
                pred_loss = F.binary_cross_entropy_with_logits(pred_logits, opp_hands)
                
                total_loss = dqn_loss + (0.5 * pred_loss)

            self.optimizer.zero_grad()
            self.scaler.scale(total_loss).backward()
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(self.policy_net.parameters(), 10.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            current_q, pred_logits = self.policy_net(states)
            current_q = current_q.gather(1, actions.unsqueeze(1)).squeeze(1)
            with torch.no_grad():
                next_q_vals = self.target_net(next_states)[0]
                next_actions = self.policy_net(next_states)[0].argmax(dim=1, keepdim=True)
                next_q = next_q_vals.gather(1, next_actions).squeeze(1)
                target_q = rewards + (1 - dones) * self.gamma * next_q
            
            td_errors = torch.abs(current_q - target_q).detach()
            dqn_loss = (weights * F.smooth_l1_loss(current_q, target_q, reduction='none')).mean()
            pred_loss = F.binary_cross_entropy_with_logits(pred_logits, opp_hands)
            total_loss = dqn_loss + (0.5 * pred_loss)
            
            self.optimizer.zero_grad()
            total_loss.backward()
            nn.utils.clip_grad_norm_(self.policy_net.parameters(), 10.0)
            self.optimizer.step()
        
        new_priorities = td_errors.cpu().numpy() + 1e-5
        memory.update_priorities(indices, new_priorities)
        return total_loss.item()

    def update_target_network(self):
        state_dict = self.policy_net.state_dict()
        new_state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
        self.target_net.load_state_dict(new_state_dict)

    def load_weights(self, path):
        """
        Robust Partial Weight Loading with ACCURATE logging.
        """
        try:
            print(f"[*] Loading weights from {path}...")
            # Try strict load first
            state_dict = torch.load(path, map_location=self.device, weights_only=True)
            self.policy_net.load_state_dict(state_dict, strict=True)
            print("[*] Perfect match! All layers loaded successfully.")
            
        except Exception as e:
            # Fallback to partial
            print(f"[!] Strict load failed ({e}). Attempting partial load...")
            try:
                state_dict = torch.load(path, map_location=self.device, weights_only=True)
                model_state = self.policy_net.state_dict()
                pretrained_dict = {
                    k: v for k, v in state_dict.items() 
                    if k in model_state and v.size() == model_state[k].size()
                }
                model_state.update(pretrained_dict)
                self.policy_net.load_state_dict(model_state)
                
                print(f"[*] Success! Loaded {len(pretrained_dict)}/{len(model_state)} layers.")
                if len(pretrained_dict) < len(model_state):
                    print("[!] Warning: Some layers were initialized randomly.")
            except Exception as e2:
                print(f"[!] Critical Error: Could not load weights at all. ({e2})")

        self.update_target_network()
        self.policy_net.eval()

    def save_snapshot(self, episode_num):
        folder = "agent/weights/snapshots"
        if not os.path.exists(folder): os.makedirs(folder, exist_ok=True)
        path = f"{folder}/snapshot_ep{episode_num}.pth"
        state_dict = self.policy_net.state_dict()
        clean_state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
        torch.save(clean_state_dict, path)
        torch.save(clean_state_dict, "agent/weights/badaamsath_dueling_v4.pth")