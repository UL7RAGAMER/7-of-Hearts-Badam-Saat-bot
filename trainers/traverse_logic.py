import torch
import random
import numpy as np
from agents.vr_pdcfr_agent import VRDeepPDCFRAgent
from game.engine import BadaamSathEngine
from game.infoset_encoder import BadaamSathEncoder

def regret_matching(advantages):
    """
    Calculates strategy from cumulative advantages.
    Fixed to handle GPU-side tensors safely.
    """
    # 1. Clip negative advantages to 0
    plus_advantages = torch.clamp(advantages, min=0)
    
    # 2. Sum the advantages (keeps the tensor on the 5070 Ti)
    sum_adv = torch.sum(plus_advantages)
    
    # 3. If sum is positive, normalize; otherwise, uniform distribution
    if sum_adv.item() > 0:
        return plus_advantages / sum_adv
    else:
        return torch.ones_like(advantages) / len(advantages)

class VRDeepPDCFRTraverser:
    def __init__(self, engine, encoder, networks, opponents, alpha=2.3):
        self.engine : BadaamSathEngine = engine
        self.encoder : BadaamSathEncoder = encoder
        self.nets : VRDeepPDCFRAgent = networks  # Contains R, r, Q, and Pi
        self.alpha = alpha
        self.opponents = opponents
    def traverse(self, t, player_i, adv_buffer, strat_buffer, hist_buffer):
        """
        Implementation of Algorithm 2 (Game Traversal)
        t: current iteration
        player_i: the 'traversing' player we are calculating regrets for
        """
        self.engine.reset()
        return self._recursive_traverse(t, player_i, adv_buffer, strat_buffer, hist_buffer, 1.0)

    def _recursive_traverse(self, t, player_i, adv_buffer, strat_buffer, hist_buffer, reach_prob):
        # 1. Base Case: Terminal Node
        if self.engine.done:
            return self.engine.get_utility(player_i)

        state = self.engine.get_state(self.engine.current_player)
        current_p = self.engine.current_player
        legal_moves = state['legal_moves']
        if current_p != player_i and self.opponents and current_p in self.opponents:
            # Use fixed strategy for opponents [cite: 211, 282]
            action = self.opponents[current_p].select_action(state)
            self.engine.apply_move(action)
            return self._recursive_traverse(t, player_i, adv_buffer, strat_buffer, hist_buffer, reach_prob)
        

        info_set = self.encoder.encode(state, self.engine.current_player).to(self.nets.device)
        

        
        # 2. Get current strategy using Regret Matching (RM)
        with torch.no_grad():
            pred_adv = self.nets.adv_nets.get_prediction(info_set, t, self.alpha)
            legal_indices = [self.encoder.action_to_idx(m) for m in legal_moves]

            # Slice the predicted advantages to ONLY include legal moves
            # This ensures 'strategy' has the same length as 'legal_moves'
            legal_advantages = pred_adv[legal_indices]
            strategy = regret_matching(legal_advantages)

        # 3. Outcome Sampling
        epsilon = 0.5 # Use 0.6 for better convergence as per Paper 1
        
        # strategy and (epsilon / len(legal_moves)) are now both size len(legal_moves)
        sampling_probs = (1 - epsilon) * strategy + (epsilon / len(legal_moves))
        
        # Convert to numpy and ensure sum is exactly 1.0 (prevents floating point errors)
        p_v = sampling_probs.cpu().numpy().astype('float64')
        p_v /= p_v.sum()

        # Now len(legal_moves) == len(p_v)
        action_idx = np.random.choice(len(legal_moves), p=p_v)
        action = legal_moves[action_idx]

        # 4. Recursion: Apply move and get utility from deeper in the tree
        self.engine.apply_move(action)
        u_tail = self._recursive_traverse(t, player_i, adv_buffer, strat_buffer, hist_buffer, 
                                          reach_prob * sampling_probs[action_idx])

        # 5. Variance Reduction & Advantage Calculation
        if state['turn'] == player_i:
            # Get baseline values from Q-network
            with torch.no_grad():
                baseline_q = self.nets.hist_q(info_set) # This is on GPU
            
            # FIX: Create this tensor directly on the GPU to match baseline_q
            sampled_adv = torch.zeros(self.nets.action_dim, device=self.nets.device)
            
            for j, move_idx in enumerate(legal_indices):
                # Ensure we only process valid indices to prevent GPU crashes
                if 0 <= move_idx < self.nets.action_dim:
                    if j == action_idx:
                        # bar_r calculation (Equation 4)
                        # We use .item() or keep as tensor; staying as tensor is faster on GPU
                        bar_r = (u_tail - baseline_q[move_idx]) / sampling_probs[j].item() + baseline_q[move_idx]
                    else:
                        bar_r = baseline_q[move_idx]
                    
                    sampled_adv[move_idx] = bar_r

            # 6. Store in Buffers
            # CRITICAL: We move back to CPU before storing in buffers to save VRAM 
            # and prevent the 5070 Ti from running out of memory during long sessions.
            adv_buffer.add(info_set.cpu(), sampled_adv.cpu())
            
            # Strategy and InfoSet should also be moved to CPU for storage
            strat_buffer.add(info_set.cpu(), t, strategy.cpu()) 
            hist_buffer.add(info_set.cpu(), action_idx, u_tail)

        return u_tail