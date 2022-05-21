# Similar to first_strategy.py, but experiments
# are run once with and once without
# information source to compare
import numpy as np
import matplotlib.pyplot as plt

from agent import Agent
from mab import MAB

trials = 1000
p_vals = np.linspace(0, 1, 41)[:-1]
N = 10

mode = 1
if mode == 0:
    payoff_per_arm = 10
    cost_per_arm = 0.5
    rate_per_arm = 3
elif mode == 1:
    payoff_per_arm = np.random.randint(1, 11, size=N)
    cost_per_arm = np.random.randint(1, 11, size=N)
    rate_per_arm = np.random.randint(1, 11, size=N)
else:
    p = np.linspace(0, 1, N + 1)
    vals = np.power(30, 1.5 * (p - 0.2)) - 1
    payoff_per_arm = (vals[1:] - vals[:-1]) / (p[1:] - p[:-1])
    cost_per_arm = -(vals[:-1] - p[:-1] * payoff_per_arm)
    rate_per_arm = np.ones(N)


def plot_values(label, priori=False):
    mab = MAB(
        N,
        payoff_per_arm,
        cost_per_arm,
        rate_per_arm,
        b_ind=False,
    )

    agent = Agent(mab, 0.5)

    data = []
    for p in p_vals:
        agent.p0 = p
        if priori:
            agent.mab.priori = p
        agent.reset()

        vals = []
        success = 0
        for _ in range(trials):
            agent.first_strategy()
            vals.append(agent.mab.value)

            if agent.mab.payoff != 0:
                success += 1

            agent.reset()
        data.append(vals)
        print(f"Successes: {success}/{trials}")

    avg_value = np.mean(data, axis=1)
    std_value = np.std(data, axis=1)
    plt.errorbar(p_vals, avg_value, yerr=std_value, fmt=".", label=label)


plot_values("misspecification", priori=False)
plot_values("priori matches initial belief", priori=True)

plt.xticks(p_vals[::2])

plt.title(
    "Agent's strategy based on Theorem 1 in Multi-Armed Exponential Bandit [Gao et al]"
)
plt.xlabel("initial belief $p$")
plt.ylabel("value")
plt.legend()
plt.show()
