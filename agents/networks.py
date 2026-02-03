import torch
import torch.nn as nn
import torch.nn.functional as F

class BadaamSathBaseNet(nn.Module):
    """
    High-capacity architecture using LayerNorm. 
    Works perfectly for both batch training and single-state traversal.
    """
    def __init__(self, input_dim, action_dim):
        super(BadaamSathBaseNet, self).__init__()
        # 1024 units per layer for ~12MB+ model size
        self.fc1 = nn.Linear(input_dim, 1024)
        self.ln1 = nn.LayerNorm(1024) 
        
        self.fc2 = nn.Linear(1024, 1024)
        self.ln2 = nn.LayerNorm(1024)
        
        self.fc3 = nn.Linear(1024, 1024)
        self.ln3 = nn.LayerNorm(1024)
        
        self.head = nn.Linear(1024, action_dim)

    def forward(self, x):
        # LayerNorm handles 1D (traversal) and 2D (training) inputs automatically
        x = F.relu(self.ln1(self.fc1(x)))
        x = F.relu(self.ln2(self.fc2(x)))
        x = F.relu(self.ln3(self.fc3(x)))
        return self.head(x)
# Note: AdvantageNetworks, HistoryValueNet, and AverageStrategyNet 
# remain the same as they wrap this base class.
class AdvantageNetworks(nn.Module):
    """
    Contains the Cumulative (R) and Instantaneous (r) networks.
    VR-DeepPDCFR+ fits cumulative advantages via bootstrapping[cite: 44, 181].
    """
    def __init__(self, input_dim, action_dim):
        super().__init__()
        # Cumulative Advantage Network (R) - Predicts bootstrapped total [cite: 116, 180]
        self.cum_adv = BadaamSathBaseNet(input_dim, action_dim)
        
        # Instantaneous Advantage Network (r) - Used for PDCFR+ predictions [cite: 49, 181]
        self.inst_adv = BadaamSathBaseNet(input_dim, action_dim)

    def get_prediction(self, state, t, alpha):
        """
        Predicts next iteration's cumulative advantages: 
        max(R(t-1), 0) * discount + r(t) [cite: 92, 183]
        """
        with torch.no_grad():
            r_prev = self.cum_adv(state)
            r_curr = self.inst_adv(state)
            discount = (t**alpha) / (t**alpha + 1)
            return torch.clamp(r_prev, min=0) * discount + r_curr

class HistoryValueNet(nn.Module):
    """
    Estimates action values (Q) at history nodes to reduce variance.
    Adopted from the DREAM approach to stabilize model-free learning[cite: 38, 46, 193].
    """
    def __init__(self, input_dim, action_dim):
        super().__init__()
        self.q_net = BadaamSathBaseNet(input_dim, action_dim)

    def forward(self, state):
        return self.q_net(state)

class AverageStrategyNet(nn.Module):
    """
    The final 'bot' network (Π) that approximates the Nash Equilibrium[cite: 81, 136, 173].
    """
    def __init__(self, input_dim, action_dim):
        super().__init__()
        self.policy_net = BadaamSathBaseNet(input_dim, action_dim)

    def forward(self, state):
        # Outputs probability logits for each card in hand [cite: 906]
        return F.softmax(self.policy_net(state), dim=-1)