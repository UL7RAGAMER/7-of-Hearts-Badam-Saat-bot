import torch
import torch.nn.functional as F

class BadaamSathBot:
    def __init__(self, adv_net, policy_net, encoder, device=None):
        self.policy_net = policy_net
        self.encoder = encoder
        self.adv_net = adv_net

        # FIX: Add explicit device handling
        if device is None:
            # Auto-detect device from policy network
            if hasattr(policy_net, 'policy_net'):
                self.device = next(policy_net.policy_net.parameters()).device
            else:
                self.device = next(policy_net.parameters()).device
        else:
            self.device = device

    def select_action(self, state):
            info_set = self.encoder.encode(state, state['turn']).to(self.device)
            
            with torch.no_grad():
                # Get raw advantage scores (how profitable each move is)
                advantages = self.adv_net(info_set)
                
                # Mask illegal moves with negative infinity so they are never picked
                legal_indices = [self.encoder.action_to_idx(m) for m in state['legal_moves']]
                mask = torch.full_like(advantages, float('-inf'))
                mask[legal_indices] = 0
                
                masked_advantages = advantages + mask
                
                # GREEDY SELECTION: Pick the single most profitable move
                # This is pure exploitation.
                action_idx = torch.argmax(masked_advantages).item()
                
            return self.encoder.idx_to_action(action_idx)
import torch
import torch.nn.functional as F

class BadaamSathExploitativeBot:
    def __init__(self, agent, encoder, use_cum_adv=True, device=None):
        self.encoder = encoder
        self.device = device if device else agent.device
        self.use_cum_adv = use_cum_adv
        
        if use_cum_adv:
            self.net = agent.adv_nets.cum_adv # The Aggressive Network
        else:
            self.net = agent.policy_net       # The Safe Network

    def select_action(self, state):
        info_set = self.encoder.encode(state, state['turn']).to(self.device)
        with torch.no_grad():
            output = self.net(info_set)
            
            legal_indices = [self.encoder.action_to_idx(m) for m in state['legal_moves']]
            mask = torch.full_like(output, float('-inf'))
            mask[legal_indices] = 0
            
            # Greedy Argmax is used for BOTH networks here
            action_idx = torch.argmax(output + mask).item()
            
        return self.encoder.idx_to_action(action_idx)
        
    def select_action_index(self, state):
            """
            NEW METHOD: Returns the raw integer index (0-52).
            Required to safely handle the 53rd 'Pass' action without crashing.
            """
            # Encoder expects: state dict, current player ID
            info_set = self.encoder.encode(state, state['turn']).to(self.device)
            
            with torch.no_grad():
                output = self.net(info_set)
                
                # Mask illegal moves
                # We must be careful: if the checkpoint has 53 actions, 
                # ensure 'legal_indices' maps correctly.
                legal_indices = [self.encoder.action_to_idx(m) for m in state['legal_moves']]
                
                # Create mask of same shape as output (likely 53)
                mask = torch.full_like(output, float('-inf'))
                
                # Safe assignment: ignore indices that might be out of bounds for this specific net
                for idx in legal_indices:
                    if idx < output.shape[0]:
                        mask[idx] = 0
                
                # EXPLOIT: Always pick the absolute best card (Argmax)
                action_idx = torch.argmax(output + mask).item()
                
            return action_idx