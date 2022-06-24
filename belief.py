"""
Author: Mara van der Meulen
---
Experiment comparing two methods to calculate the agent's belief,
this belief represents the probability of a breakthrough occuring in
the future given that none has occured yet.
"""
import numpy as np
import matplotlib.pyplot as plt

from agent import Agent
from mab import MAB
import arm_settings as arm

trials = 200
p0 = 0.95

# MAB settings
N = 20

# Choose mode "identical", "random", "specific"
arm_mode = "specific"
if arm_mode == "identical":
    payoff_per_arm, cost_per_arm, rate_per_arm = arm.setting_a()
elif arm_mode == "random":
    payoff_per_arm, cost_per_arm, rate_per_arm = arm.setting_b(N)
else:
    payoff_per_arm, cost_per_arm, rate_per_arm = arm.setting_c(N)


def plot_belief(dt, payoff_per_arm, cost_per_arm, rate_per_arm):
    """
    Plot a comparison between two methods to calculate belief for
    a specified time step size and model parameters.
    """
    mab = MAB(N, payoff_per_arm, cost_per_arm, rate_per_arm, b_ind=False)
    agent = Agent(mab, p0)

    # Set time step
    agent.mab.dt = dt

    # Compute belief using bayesian updates and adaptations
    p_bay = []
    p_upd = []
    for _ in range(trials):
        agent.first_strategy()
        p_bay.append(agent.p_bay)
        p_upd.append(agent.p_upd)
        agent.reset()

    # PLOT the maximal value over all runs (deterministic except for b=0 or 1)
    length = np.max([len(p) for p in p_bay])
    p_bay = np.array([np.pad(r, (0, length - len(r))) for r in p_bay])
    p_upd = np.array([np.pad(r, (0, length - len(r))) for r in p_upd])

    plt.plot(
        np.arange(np.shape(p_bay)[1]) * agent.mab.dt,
        np.max(p_bay, axis=0),
        label="Bayesian updates",
    )
    plt.plot(
        np.arange(np.shape(p_upd)[1]) * agent.mab.dt,
        np.max(p_upd, axis=0),
        label="adaptations",
    )

    plt.xlim(0, 4.3)
    plt.ylim(0, 1)

    plt.title(f"$dt$ = {dt}")
    plt.xlabel("time $t$")
    plt.ylabel("belief $p$")
    plt.legend()


def belief_figure(payoff_per_arm, cost_per_arm, rate_per_arm):
    """
    Plot a comparison of the computed beliefs for several time step sizes.
    """
    _ = plt.subplots(4, figsize=(10, 7))

    vals = [0.2, 0.1, 0.05, 0.01]

    for i in range(4):
        plt.subplot(2, 2, i + 1)
        plot_belief(vals[i], payoff_per_arm, cost_per_arm, rate_per_arm)

    plt.suptitle(
        "Comparison of different methods to calculate the agent's belief over time."
    )
    plt.tight_layout()
    plt.show()


belief_figure(payoff_per_arm, cost_per_arm, rate_per_arm)
