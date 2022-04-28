from urllib.request import proxy_bypass
import numpy as np
import matplotlib.pyplot as plt

from agent import Agent
from mab import MAB

trials = 200
p0 = 0.8

# MAB settings
N = 10

payoff_per_arm = 10
cost_per_arm = 0.5
rate_per_arm = 3

# ARMs_payoff = payoff_per_arm * np.ones(N)
# ARMs_cost = cost_per_arm * np.ones(N)
# ARMs_rate = rate_per_arm * np.ones(N)


def plot_belief(dt):
    mab = MAB(N, payoff_per_arm, cost_per_arm, rate_per_arm)
    mab.arm_success = np.resize(1, N)
    agent = Agent(mab, p0)

    # Set time step
    agent.mab.dt = dt

    # Compute belief using bayesian updates and adaptations
    p_bay = []
    p_upd = []
    for _ in range(trials):
        agent.first_strategy(agent.bay_updates)
        p_bay.append(agent.p_bay)
        p_upd.append(agent.p_upd)
        agent.reset()

    q = np.argmax([len(r) for r in p_bay])
    p_bay_l = p_bay[q]
    p_upd_l = p_upd[q]

    plt.plot(
        np.arange(len(p_bay_l)) * agent.mab.dt,
        p_bay_l,
        label="bayesian updates",
    )
    plt.plot(
        np.arange(len(p_upd_l)) * agent.mab.dt,
        p_upd_l,
        label="adaptations",
    )

    # p_bay = np.array(
    #     [
    #         np.pad(r, (0, np.max([len(p) for p in p_bay]) - len(r)))
    #         for r in p_bay
    #     ]
    # )
    # p_upd = np.array(
    #     [
    #         np.pad(r, (0, np.max([len(p) for p in p_upd]) - len(r)))
    #         for r in p_upd
    #     ]
    # )

    # plt.plot(
    #     np.arange(np.shape(p_bay)[1]) * agent.mab.dt,
    #     np.mean(p_bay, axis=0),
    #     label="bayesian updates",
    # )
    # plt.plot(
    #     np.arange(np.shape(p_upd)[1]) * agent.mab.dt,
    #     np.mean(p_upd, axis=0),
    #     label="adaptations",
    # )

    # plt.errorbar(
    #     np.arange(np.shape(p_bay)[1]) * agent.mab.dt,
    #     np.mean(p_bay, axis=0),
    #     yerr=np.std(p_bay, axis=0),
    #     fmt=".",
    #     color="blue",
    #     label="bayesian updates",
    # )
    # plt.errorbar(
    #     np.arange(np.shape(p_upd)[1]) * agent.mab.dt,
    #     np.mean(p_upd, axis=0),
    #     yerr=np.std(p_upd, axis=0),
    #     fmt=".",
    #     color="red",
    #     label="adaptations",
    # )

    plt.xlim(0, 1.75)
    plt.ylim(0, p0)

    plt.title(f"$dt$ = {dt}")
    plt.xlabel("time $t$")
    plt.ylabel("belief $p$")
    plt.legend()


fig = plt.figure(figsize=(15, 20))

plt.subplot(2, 2, 1)
plot_belief(0.2)

plt.subplot(2, 2, 2)
plot_belief(0.1)

plt.subplot(2, 2, 3)
plot_belief(0.05)

plt.subplot(2, 2, 4)
plot_belief(0.01)

plt.suptitle(
    "Comparison of different methods to calculate the agent's belief over time."
)

plt.show()
