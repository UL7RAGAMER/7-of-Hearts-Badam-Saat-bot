import numpy as np
import csv
import os

class BadaamSathMetrics:
    def __init__(self, log_file="training_log.csv"):
        self.iteration_losses = {'R': [], 'r': [], 'Q': [], 'Pi': []}
        self.rewards = []
        self.log_file = log_file
        
        # Initialize the CSV file with headers if it doesn't exist
        if not os.path.exists(self.log_file):
            with open(self.log_file, mode='w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['epoch', 'loss_R', 'loss_r', 'loss_Q', 'loss_Pi', 'avg_reward'])

    def record_loss(self, net_type, loss_val):
        self.iteration_losses[net_type].append(loss_val)

    def report(self, epoch):
        # Calculate recent averages (last 10 epochs)
        window = 10
        avg_R = np.mean(self.iteration_losses['R'][-window:]) if self.iteration_losses['R'] else 0
        avg_r = np.mean(self.iteration_losses['r'][-window:]) if self.iteration_losses['r'] else 0
        avg_Q = np.mean(self.iteration_losses['Q'][-window:]) if self.iteration_losses['Q'] else 0
        avg_Pi = np.mean(self.iteration_losses['Pi'][-window:]) if self.iteration_losses['Pi'] else 0
        avg_reward = np.mean(self.rewards[-window:]) if self.rewards else 0

        # 1. Print to Console for immediate feedback
        header = f"--- [ EPOCH {epoch:04d} ] ---"
        print(header)
        print(f"{'Regret (R)':<15} : {avg_R:>10.4f}")
        print(f"{'Value (Q)':<15} : {avg_Q:>10.4f}")
        print(f"{'Strategy (Pi)':<15} : {avg_Pi:>10.8f}")
        print(f"{'Avg Payoff':<15} : {avg_reward:>10.2f}")
        print("-" * len(header) + "\n")

        # 2. Append to CSV for persistent logging
        with open(self.log_file, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch, avg_R, avg_r, avg_Q, avg_Pi, avg_reward])