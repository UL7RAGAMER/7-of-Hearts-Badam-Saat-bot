from game.engine import BadaamSathEngine
from game.infoset_encoder import BadaamSathEncoder
from agents.vr_pdcfr_agent import VRDeepPDCFRAgent
from memory.advantage_buffer import AdvantageBuffer
from memory.reservoir_buffer import ReservoirBuffer
from memory.history_buffer import HistoryBuffer
from trainers.algorithm_1 import VRDeepPDCFRTrainer

def train():
    # 1. Setup Environment
    engine = BadaamSathEngine()
    encoder = BadaamSathEncoder()
    
    # 2. Setup VR-DeepPDCFR+ Agent
    # Hyperparameters from Paper 1: alpha=2.3, gamma=2.0 
    agent = VRDeepPDCFRAgent(
        input_dim=encoder.input_dim, 
        action_dim=encoder.action_dim,
        alpha=2.0, 
        gamma=3.0
    )

    # 3. Setup Buffers
    # Buffer sizes used in Paper 1: 1M for Adv, Strategy, and History 
    adv_buffer = AdvantageBuffer(capacity=1000000)
    strat_buffer = ReservoirBuffer(capacity=1000000)
    hist_buffer = HistoryBuffer(capacity=1000000)

    # 4. Initialize Trainer (Algorithm 1)
    trainer = VRDeepPDCFRTrainer(
        agent=agent,
        engine=engine,
        encoder=encoder,
        adv_buffer=adv_buffer,
        strat_buffer=strat_buffer,
        hist_buffer=hist_buffer
    )

    # 5. Start Training
    # Paper 1 suggests 10,000 traversals per iteration 
    trainer.train(num_iterations=1000, episodes_per_iter=1000)
