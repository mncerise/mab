"""
Author: Mara van der Meulen
---
This file contains several parameter settings for the Multi-Armed Bandits.
"""
import numpy as np


def setting_a():
    """
    Define parameters for identical arms.
    """
    payoff_per_arm = 10
    cost_per_arm = 6
    rate_per_arm = 3

    return (payoff_per_arm, cost_per_arm, rate_per_arm)


def setting_b(N):
    """
    Define parameters for random arms.
    """
    payoff_per_arm = np.random.randint(1, 11, size=N)
    cost_per_arm = np.random.randint(1, 11, size=N)
    rate_per_arm = np.random.randint(1, 11, size=N)

    return (payoff_per_arm, cost_per_arm, rate_per_arm)


def setting_c(N):
    """
    Define parameters for specific arms, explained and illustrated
    in Section 3.1 in the accompanying thesis.
    """
    p = np.linspace(0, 1, N + 1)
    vals = np.power(30, 1.5 * (p - 0.2)) - 1
    payoff_per_arm = np.maximum(0, (vals[1:] - vals[:-1]) / (p[1:] - p[:-1]))
    cost_per_arm = np.maximum(0, -(vals[:-1] - p[:-1] * payoff_per_arm))
    rate_per_arm = np.ones(N)

    return (payoff_per_arm, cost_per_arm, rate_per_arm)
