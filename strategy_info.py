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

p = np.linspace(0, 1, N + 1)
vals = np.power(30, 1.5 * (p - 0.2)) - 1
payoff_per_arm = (vals[1:] - vals[:-1]) / (p[1:] - p[:-1])
cost_per_arm = -(vals[:-1] - p[:-1] * payoff_per_arm)
rate_per_arm = np.ones(N)


def plot_values(label, info=False):
    mab = MAB(
        N,
        payoff_per_arm,
        cost_per_arm,
        rate_per_arm,
        info=info,
        source=(np.max(payoff_per_arm - cost_per_arm / rate_per_arm) * 0.2, 1),
        b_ind=False,
    )

    agent = Agent(mab, 0.5)

    data = []
    for p in p_vals:
        agent.p0 = p
        # agent.mab.priori = p
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


plot_values("without information", info=False)
plot_values("information source", info=True)

plt.xticks(p_vals[::2])

plt.title(
    "Agent's strategy based on Theorem 1 in Multi-Armed Exponential Bandit [Gao et al]"
)
plt.xlabel("initial belief $p$")
plt.ylabel("value")
plt.legend()
plt.show()
