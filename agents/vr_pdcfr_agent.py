import copy
import torch
import torch.nn as nn
import torch.optim as optim
from agents.networks import AdvantageNetworks, HistoryValueNet, AverageStrategyNet
from torch.optim.lr_scheduler import ReduceLROnPlateau
class VRDeepPDCFRAgent:
    def __init__(self, input_dim, action_dim, alpha=2.3, gamma=2.0):
        # Hyperparameters specific to VR-DeepPDCFR+ [cite: 219]
        self.alpha = alpha
        self.gamma = gamma
        self.action_dim = action_dim
        self.device = device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize the four required networks [cite: 116, 117, 136]
        self.adv_nets = AdvantageNetworks(input_dim, action_dim).to(device) # R and r
        self.hist_q = HistoryValueNet(input_dim, action_dim).to(device)     # Q
        self.policy_net = AverageStrategyNet(input_dim, action_dim).to(device) # Π 

        # Optimizers (Standard Adam with 0.001 lr as per paper) [cite: 485]
        self.opt_R = optim.Adam(self.adv_nets.cum_adv.parameters(), lr=0.001)
        self.opt_r = optim.Adam(self.adv_nets.inst_adv.parameters(), lr=0.001)
        self.opt_Q = optim.Adam(self.hist_q.parameters(), lr=0.001)
        self.opt_Pi = optim.Adam(self.policy_net.parameters(), lr=0.001)

        self.sched_R = ReduceLROnPlateau(self.opt_R, mode='min', factor=0.1, patience=100,min_lr=1e-3)
        self.sched_r = ReduceLROnPlateau(self.opt_r, mode='min', factor=0.1, patience=100,min_lr=1e-3)
        self.target_cum_adv = copy.deepcopy(self.adv_nets.cum_adv).to(self.device)
        self.target_update_freq = 100 # Sync every 20 iterations
        
    def update_advantage_networks(self, t, batch):
        """Implements the bootstrapped loss for R and r[cite: 130, 133]."""
        states_list, adv_list = zip(*batch) # target_advs are the sampled advantages (bar_r)
        
        states = torch.stack(states_list).to(self.device)
        target_advs = torch.stack(adv_list).to(self.device)
        # 1. Update R (Cumulative Advantage Network) [cite: 130]
        # Loss follows: (max(R(t-1), 0) * discount + bar_r - R(t))^2
        with torch.no_grad():
            # USE TARGET NETWORK HERE for stability
            r_prev = self.target_cum_adv(states) 
            discount = (t**self.alpha) / ((t**self.alpha + 1))
            r_target = torch.clamp(r_prev, min=0) * discount + target_advs

        r_current_pred = self.adv_nets.cum_adv(states)
        loss_R = nn.MSELoss()(r_current_pred, r_target)
        
        self.opt_R.zero_grad()
        loss_R.backward()
        self.opt_R.step()

        # 2. Update r (Instantaneous Advantage Network) [cite: 133, 182]
        # Fits the current sampled advantages for predictive steps
        r_inst_pred = self.adv_nets.inst_adv(states)
        loss_r = nn.MSELoss()(r_inst_pred, target_advs)
        
        self.opt_r.zero_grad()
        loss_r.backward()
        self.opt_r.step()
        return loss_R.item(), loss_r.item()
    def update_average_policy(self, t, T, batch):
        """Trains the bot (Π) using weighted strategy samples."""
        # 1. Unzip the batch of samples
        # Assuming batch is [(state, iter_t, strategy), ...]
        states_list, iter_ts_list, target_strategies_list = zip(*batch)
        
        # 2. Convert to tensors and move to device
        states = torch.stack(states_list).to(self.device)
        iter_ts = torch.tensor(iter_ts_list, device=self.device, dtype=torch.float32)
        padded_strategies = []
        for strat in target_strategies_list:
            # Create a zero tensor of the full action dimension
            pad = torch.zeros(self.action_dim, device=self.device)
            # Fill in the legal action probabilities
            pad[:strat.size(0)] = strat
            padded_strategies.append(pad)
        target_strategies = torch.stack(padded_strategies).to(self.device)

        # 3. Apply predictive weighting: (t/T)^gamma
        # We use unsqueeze(1) to ensure the weights broadcast correctly across the action dimension
        weights = ((iter_ts / T)**self.gamma).unsqueeze(1)
        
        pred_pi = self.policy_net(states)
        
        # 4. Weighted MSE loss to fit the strategies
        loss = torch.mean(weights * torch.sum((pred_pi - target_strategies)**2, dim=1))
        
        self.opt_Pi.zero_grad()
        loss.backward()
        self.opt_Pi.step()

        return loss.item()
    def update_history_network(self, batch):
        """
        Trains Q to be a baseline for variance reduction.
        """
        # 1. Unzip the list of tuples into separate lists
        states_list, action_list, utility_list = zip(*batch)
        
        # 2. Convert to tensors and move to device
        states = torch.stack(states_list).to(self.device)
        # Action indices and utilities are typically scalars per sample
        action_indices = torch.tensor(action_list, device=self.device)
        utilities = torch.tensor(utility_list, device=self.device, dtype=torch.float32)
        
        # 3. Predict values for the actions taken
        current_q_values = self.hist_q(states)
        
        # 4. Create targets
        targets = current_q_values.clone().detach()
        
        # Optimized batch update using scatter or advanced indexing
        # This replaces the manual loop for better performance
        row_indices = torch.arange(len(action_indices), device=self.device)
        targets[row_indices, action_indices] = utilities
            
        loss_Q = nn.MSELoss()(current_q_values, targets)
        
        self.opt_Q.zero_grad()
        loss_Q.backward()
        self.opt_Q.step()

        return loss_Q.item()
    
    def step_schedulers(self, metrics):
        """Updates the learning rates based on recent performance."""
        # Use the latest R-loss to decide if we should decay
        if 'R' in metrics and metrics['R']:
            self.sched_R.step(metrics['R'][-1])
            self.sched_r.step(metrics['R'][-1])