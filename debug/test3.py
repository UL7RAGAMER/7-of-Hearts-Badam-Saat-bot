import torch
import random
import numpy as np

from game.engine import BadaamSathEngine
from game.infoset_encoder import BadaamSathEncoder
from agents.vr_pdcfr_agent import VRDeepPDCFRAgent


# ----------------------------
# Utility
# ----------------------------
def load_agent(checkpoint_path, encoder, device="cpu"):
    agent = VRDeepPDCFRAgent(
        input_dim=encoder.input_dim,
        action_dim=encoder.action_dim,
    )

    # Move ONLY the networks
    agent.device = torch.device(device)
    agent.policy_net.to(agent.device)

    # Load policy weights
    agent.policy_net.load_state_dict(
        torch.load(checkpoint_path, map_location=agent.device)
    )
    agent.policy_net.eval()

    # Freeze everything (Phase 0)
    for p in agent.policy_net.parameters():
        p.requires_grad = False

    return agent



def sample_policy_action(agent, encoder, state, p_id, legal_moves):
    obs = encoder.encode(state, p_id)

    # ---- FIX: handle tensor vs numpy safely ----
    if isinstance(obs, torch.Tensor):
        obs_tensor = obs.detach().clone().float().to(agent.device)
    else:
        obs_tensor = torch.from_numpy(obs).float().to(agent.device)
    # --------------------------------------------


    with torch.no_grad():
        logits = agent.policy_net(obs_tensor)

        legal_indices = torch.tensor(
            [encoder.action_to_idx(m) for m in legal_moves],
            device=agent.device
        )

        legal_logits = logits[legal_indices]
        probs = legal_logits

        idx = torch.multinomial(probs, 1).item()
        return legal_moves[idx]


# ----------------------------
# Phase-0 Eval A: Self-Play
# ----------------------------

def self_play_eval(agent, encoder, games=1000):
    engine = BadaamSathEngine()
    seat_payoffs = {i: [] for i in range(4)}

    for _ in range(games):
        engine.reset()
        done = False

        while not done:
            p_id = engine.current_player
            state = engine.get_state(p_id)
            legal_moves = engine.get_legal_moves(p_id)

            action = sample_policy_action(
                agent, encoder, state, p_id, legal_moves
            )
            done = engine.apply_move(action)

        # collect payoff for ALL seats
        for i in range(4):
            seat_payoffs[i].append(engine.get_utility(i))

    stats = {}
    for i in range(4):
        arr = np.asarray(seat_payoffs[i])
        stats[f"seat_{i}"] = {
            "mean": float(arr.mean()),
            "std": float(arr.std()),
        }

    return stats

def eval_vs_random(agent, encoder, games=1000):
    engine = BadaamSathEngine()

    wins = 0
    scores = []

    for _ in range(games):
        engine.reset()
        done = False

        while not done:
            p_id = engine.current_player
            state = engine.get_state(p_id)
            legal_moves = engine.get_legal_moves(p_id)

            if p_id == 0:
                action = sample_policy_action(
                    agent, encoder, state, p_id, legal_moves
                )
            else:
                action = random.choice(legal_moves)

            done = engine.apply_move(action)

        utilities = [engine.get_utility(i) for i in range(4)]
        scores.append(utilities[0])

        if utilities[0] == max(utilities):
            wins += 1

    scores = np.asarray(scores)

    return {
        "win_rate": wins / games,
        "avg_payoff": float(scores.mean()),
        "best": float(scores.max()),
        "worst": float(scores.min()),
    }


# ----------------------------
# Main
# ----------------------------

if __name__ == "__main__":
    CHECKPOINT = "checkpoints/pdcfr_iter_1000.pth"
    GAMES = 1000

    encoder = BadaamSathEncoder()
    agent = load_agent(CHECKPOINT, encoder, device="cpu")

    print("\n=== Phase-0 Self-Play Evaluation (π vs π) ===")
    sp = self_play_eval(agent, encoder, games=GAMES)
    for seat, stats in sp.items():
        print(f"{seat}:")
        print(f"  mean: {stats['mean']:.4f}")
        print(f"  std : {stats['std']:.4f}")
    

    print("\n=== Sanity Check: π vs Random (NOT success metric) ===")
    rnd = eval_vs_random(agent, encoder, games=GAMES)
    for k, v in rnd.items():
        print(f"{k:>10}: {v:.4f}")
    