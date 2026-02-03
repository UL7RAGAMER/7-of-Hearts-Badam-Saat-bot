import torch
import numpy as np
from tqdm import tqdm

# Import your existing game components
from agents.strategy_wrapper import BadaamSathExploitativeBot
from game.engine import BadaamSathEngine
from game.infoset_encoder import BadaamSathEncoder
from agents.vr_pdcfr_agent import VRDeepPDCFRAgent
from agents.stratergy_based import CandidStatistician, LooseAggressive, TightPassive

# --- INSERT THE NEW CLASS HERE (Or import it if you saved it) ---

def run_evaluation(num_games=1000, model_path="checkpoints/pdcfr_iter_900.pth"):
    print(f"--- Starting Exploitation Check ({num_games} games) ---")
    
    # 1. Setup Environment
    engine = BadaamSathEngine()
    encoder = BadaamSathEncoder()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Ensure dimensions match your training (likely 412 in, 52 out)
    agent = VRDeepPDCFRAgent(input_dim=112, action_dim=53) 
    
    # 2. Robust Loading Logic
    try:
        checkpoint = torch.load(model_path)
        
        # Check if this is the "new style" full checkpoint
        if isinstance(checkpoint, dict) and 'cum_adv' in checkpoint:
            print("Found FULL agent checkpoint. Loading Advantage Networks for maximum exploitation...")
            # Load the Advantage Network (The "Killer" Brain)
            agent.adv_nets.cum_adv.load_state_dict(checkpoint['cum_adv'])
            
            # Load Policy Net just in case, though we won't use it for the exploit
            agent.policy_net.load_state_dict(checkpoint['policy_net'])
            
            # Set mode to use the Advantage Network
            use_advantage_exploit = True
            
        else:
            # Fallback for old checkpoints that were just the policy weights
            print("Found simple Policy checkpoint. Loading Policy Net...")
            agent.policy_net.load_state_dict(checkpoint)
            use_advantage_exploit = False

        agent.adv_nets.cum_adv.to(device)
        agent.policy_net.to(device)
        
    except Exception as e:
        print(f"Loading failed: {e}")
        return

    # 3. Instantiate the Bot based on what we found
    if use_advantage_exploit:
        # STRONGER: Uses cumulative regret to pick the absolute best historic move
        hero_bot = BadaamSathExploitativeBot(agent, encoder, use_cum_adv=True)
        print("Using Advantage Network for Exploitative Play.")
    else:
        # WEAKER: Uses the average strategy (Nash Equilibrium approximation)
        hero_bot = BadaamSathExploitativeBot(agent, encoder, use_cum_adv=False)
        print("Using Average Strategy for Play.")

    
    opponents = {
        1: LooseAggressive(),   #
        2: TightPassive(),      #
        3: CandidStatistician() #
    }

    # 4. Game Loop
    hero_scores = []
    wins = 0

    for i in tqdm(range(num_games)):
        engine.reset()
        
        while not engine.done:
            state = engine.get_state(engine.current_player)
            curr_player = engine.current_player
            
            if curr_player == 0:
                action = hero_bot.select_action(state)
            else:
                action = opponents[curr_player].select_action(state)
            
            engine.apply_move(action)
        
        # Game Over - Record Utility for Player 0
        utility = engine.get_utility(0)
        hero_scores.append(utility)
        if utility == 0: # Assuming positive utility means winning/beating average
            wins += 1

    # 5. Report
    avg_score = np.mean(hero_scores)
    win_rate = (wins / num_games) * 100
    
    print("\n--- Results ---")
    print(f"Hero (Exploitative) vs Heuristic Bots")
    print(f"Games Played: {num_games}")
    print(f"Win Rate:     {win_rate:.2f}%")
    print(f"Avg Utility:  {avg_score:.4f}")
    
    if avg_score > 0:
        print("\nSUCCESS: The bot is effectively exploiting the opponents.")
    else:
        print("\nFAIL: The bot is still losing on average. Consider training longer or checking the 'Pass' penalty.")

if __name__ == "__main__":
    # Make sure to point this to your actual best checkpoint file
    run_evaluation(num_games=10000, model_path="checkpoints/pdcfr_iter_900.pth")