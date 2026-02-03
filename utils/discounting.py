import torch

def get_advantage_discount(t, alpha):
    """Calculates the (t^alpha / (t^alpha + 1)) factor for cumulative regrets."""
    if t <= 1: return 0.0
    return (t**alpha) / (t**alpha + 1)

def get_strategy_weight(t, T, gamma):
    """Calculates (t/T)^gamma for linear weighting of the average strategy."""
    return (t / T)**gamma