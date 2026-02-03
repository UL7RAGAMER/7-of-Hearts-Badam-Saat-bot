# ======================
# BOT MANAGER
# ======================
import os
import sys
import torch
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agents.strategy_wrapper import BadaamSathExploitativeBot
from agents.vr_pdcfr_agent import VRDeepPDCFRAgent
from game.infoset_encoder import BadaamSathEncoder    
from utils.helper_functions import  from_bot_card   

class Bot:
    def __init__(self):
        self.ok = False

        try:
            self.encoder = BadaamSathEncoder()
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.agent = VRDeepPDCFRAgent(input_dim=112, action_dim=53) 
            
            chk_path = "../checkpoints/pdcfr_iter_1000.pth" 
            
            if os.path.exists(chk_path):
                checkpoint = torch.load(chk_path, map_location=self.device)
                use_cum_adv = False
                if isinstance(checkpoint, dict) and 'cum_adv' in checkpoint:
                    print("Bot: Loading Full Agent (High Performance)")
                    self.agent.adv_nets.cum_adv.load_state_dict(checkpoint['cum_adv'])
                    use_cum_adv = True
                elif isinstance(checkpoint, dict) and 'policy_net' in checkpoint:
                     print("Bot: Loading Policy Net (Standard)")
                     self.agent.policy_net.load_state_dict(checkpoint['policy_net'])
                else:
                    self.agent.policy_net.load_state_dict(checkpoint)

                self.agent.adv_nets.cum_adv.to(self.device)
                self.agent.policy_net.to(self.device)

                self.brain = BadaamSathExploitativeBot(
                    self.agent, 
                    self.encoder, 
                    use_cum_adv=use_cum_adv, 
                    device=self.device
                )
                self.ok = True
                print("Bot: Ready to destroy (112-dim mode).")
            else:
                print(f"Bot: Checkpoint not found at {chk_path}")
                
        except Exception as e:
            print(f"Bot Error: {e}")
            import traceback
            traceback.print_exc()

    def move(self, state_obj):
        if not self.ok: return None
        
        legal_ui = state_obj.legal_moves(state_obj.my_hand)
        if not legal_ui: return None 
        if len(legal_ui) == 1: return legal_ui[0]

        try:
            bot_state = state_obj.get_bot_state()
            action_idx = self.brain.select_action_index(bot_state)
            return from_bot_card(action_idx)
        except Exception as e:
            print(f"Inference Error: {e}")
            return None

