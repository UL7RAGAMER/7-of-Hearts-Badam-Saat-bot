import random
import torch

from agents.stratergy_based import CandidStatistician, LooseAggressive, TightPassive
from agents.vr_pdcfr_agent import VRDeepPDCFRAgent
from game.engine import BadaamSathEngine
from game.infoset_encoder import BadaamSathEncoder
from memory.advantage_buffer import AdvantageBuffer
from memory.history_buffer import HistoryBuffer
from memory.reservoir_buffer import ReservoirBuffer
from utils.metrics import BadaamSathMetrics

class VRDeepPDCFRTrainer:
    def __init__(self, agent, engine, encoder, adv_buffer, strat_buffer, hist_buffer):
        self.agent : VRDeepPDCFRAgent = agent
        self.engine : BadaamSathEngine = engine
        self.encoder : BadaamSathEncoder = encoder
        self.adv_buffer : AdvantageBuffer = adv_buffer
        self.strat_buffer : ReservoirBuffer = strat_buffer
        self.hist_buffer : HistoryBuffer = hist_buffer
        self.opponents = {
                1: CandidStatistician(),
                2: LooseAggressive(),
                3: TightPassive()
            }
            # Import traverser here to avoid circular imports
        from trainers.traverse_logic import VRDeepPDCFRTraverser
        self.traverser = VRDeepPDCFRTraverser(engine, encoder, agent, opponents=self.opponents)
        self.metrics = BadaamSathMetrics()  # Initialize the metrics tracker

    def train(self, num_iterations, episodes_per_iter):
        strategy_list = [CandidStatistician(), LooseAggressive(), TightPassive()]
        for t in range(1, num_iterations + 1):
            self.adv_buffer.clear()

            # 1. Data Collection
            for k in range(episodes_per_iter):
                shuffled_opponents = random.sample(strategy_list, len(strategy_list))
            
            # 2. Update the traverser's opponent mapping
                # Player 0 is always your learning bot
                self.traverser.opponents = {
                    1: shuffled_opponents[0],
                    2: shuffled_opponents[1],
                    3: shuffled_opponents[2]
                }
                player_i = 0
                # During traversal, the engine returns utilities at terminal nodes
                # You can capture these to populate self.metrics.rewards
                utility = self.traverser.traverse(t, player_i, self.adv_buffer, self.strat_buffer, self.hist_buffer)
                self.metrics.rewards.append(utility)
            # 2. Train and Record Metrics
            adv_batch = self.adv_buffer.sample(batch_size=2048*8)
            if adv_batch:
                l_R, l_r = self.agent.update_advantage_networks(t, adv_batch)
                self.metrics.record_loss('R', l_R)
                self.metrics.record_loss('r', l_r)

            hist_batch = self.hist_buffer.sample(batch_size=2048*8)
            if hist_batch:
                l_Q = self.agent.update_history_network(hist_batch)
                self.metrics.record_loss('Q', l_Q)

            strat_batch = self.strat_buffer.sample(batch_size=2048*8)
            if strat_batch:
                l_Pi = self.agent.update_average_policy(t, num_iterations, strat_batch)
                self.metrics.record_loss('Pi', l_Pi)
            
            self.agent.step_schedulers(self.metrics.iteration_losses)

            # 3. Print the Report
            if t % 1 == 0:  # Report every iteration
                current_lr = self.agent.opt_R.param_groups[0]['lr']
                self.metrics.report(t)
                print(f"  Learning Rate (R, r): {current_lr:.6f}")

            if t % 100 == 0:
                self.save_checkpoint(t)
            if t % self.agent.target_update_freq == 0:
                self.agent.target_cum_adv.load_state_dict(self.agent.adv_nets.cum_adv.state_dict())

    def save_checkpoint(self, iteration):
        path = f"checkpoints/pdcfr_iter_{iteration}.pth"
        
        # FIX: Save the entire agent state, not just the policy
        torch.save({
            'policy_net': self.agent.policy_net.state_dict(),
            'cum_adv': self.agent.adv_nets.cum_adv.state_dict(),
            'inst_adv': self.agent.adv_nets.inst_adv.state_dict(),
            'hist_q': self.agent.hist_q.state_dict(),
        }, path)
        
        print(f"Checkpoint (Full Agent) saved to {path}")