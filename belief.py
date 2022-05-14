from urllib.request import proxy_bypass
import numpy as np
import matplotlib.pyplot as plt

from agent import Agent
from mab import MAB

trials = 200
p0 = 0.9

# MAB settings
N = 10

payoff_per_arm = 10
cost_per_arm = 0.5
rate_per_arm = 3


def plot_belief(dt, random=False):
    mab = MAB(N, payoff_per_arm, cost_per_arm, rate_per_arm, b_ind=True)
    agent = Agent(mab, p0)

    # Set time step
    agent.mab.dt = dt

    # If random is set, arms aren't identical
    if random:
        agent.mab.arm_payoff = np.random.randint(1, 11, size=N)
        agent.mab.arm_cost = np.random.randint(1, 11, size=N)
        agent.mab.arm_rate = np.random.randint(1, 11, size=N)

    # Compute belief using bayesian updates and adaptations
    p_bay = []
    p_upd = []
    for _ in range(trials):
        agent.first_strategy()
        p_bay.append(agent.p_bay)
        p_upd.append(agent.p_upd)
        agent.reset()

    q = np.argmax([len(r) for r in p_bay])
    p_bay_l = p_bay[q]
    p_upd_l = p_upd[q]

    # PLOT the longest single run
    # plt.plot(
    #     np.arange(len(p_bay_l)) * agent.mab.dt,
    #     p_bay_l,
    #     label="Bayesian updates",
    # )
    # plt.plot(
    #     np.arange(len(p_upd_l)) * agent.mab.dt,
    #     p_upd_l,
    #     label="adaptations",
    # )

    # PLOT the average over all runs
    length = np.max([len(p) for p in p_bay])
    p_bay = np.array([np.pad(r, (0, length - len(r))) for r in p_bay])
    p_upd = np.array([np.pad(r, (0, length - len(r))) for r in p_upd])

    plt.plot(
        np.arange(np.shape(p_bay)[1]) * agent.mab.dt,
        np.mean(p_bay, axis=0),
        label="bayesian updates",
    )
    plt.plot(
        np.arange(np.shape(p_upd)[1]) * agent.mab.dt,
        np.mean(p_upd, axis=0),
        label="adaptations",
    )

    plt.errorbar(
        np.arange(np.shape(p_bay)[1]) * agent.mab.dt,
        np.mean(p_bay, axis=0),
        yerr=np.std(p_bay, axis=0),
        fmt=".",
        color="blue",
        capsize=2,
    )
    plt.errorbar(
        np.arange(np.shape(p_upd)[1]) * agent.mab.dt,
        np.mean(p_upd, axis=0),
        yerr=np.std(p_upd, axis=0),
        fmt=".",
        color="red",
        capsize=2,
    )

    plt.xlim(0, 0.6)
    plt.ylim(0, p0)

    plt.title(f"$dt$ = {dt}")
    plt.xlabel("time $t$")
    plt.ylabel("belief $p$")
    plt.legend()


def belief_figure(random=False):
    fig = plt.figure(figsize=(15, 20))

    vals = [0.2, 0.1, 0.05, 0.01]

    for i in range(4):
        plt.subplot(2, 2, i + 1)
        plot_belief(vals[i], random)

    plt.suptitle(
        "Comparison of different methods to calculate the agent's belief over time."
    )

    plt.show()


belief_figure(random=True)
