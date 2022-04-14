import numpy as np
import matplotlib.pyplot as plt

from agent import Agent
from mab import MAB

# Number of iterations
trials = 100
p_vals = np.linspace(0, 1, 11)[:-1]

# MAB settings
N = 15

payoff_per_arm = 10
cost_per_arm = 0.5
rate_per_arm = 3

ARMs_payoff = payoff_per_arm * np.ones(N)
ARMs_cost = cost_per_arm * np.ones(N)
ARMs_rate = rate_per_arm * np.ones(N)


mab = MAB(N, payoff_per_arm, cost_per_arm, rate_per_arm)
agent = Agent(mab, 0.5)

data = []
for p in p_vals:
    agent.p0 = p
    agent.p = p

    vals = []
    success = 0
    for t in range(trials):
        agent.first_strategy()
        vals.append(agent.mab.value)

        if agent.mab.payoff != 0:
            success += 1

        agent.reset()
    data.append(vals)
    print(f"Successes: {success}/{trials}")

# print(data)

avg_value = np.mean(data, axis=1)
std_value = np.std(data, axis=1)
plt.errorbar(p_vals, avg_value, yerr=std_value, fmt=".", color="red")

plt.xticks(p_vals)

plt.title(
    "Agent's strategy based on Theorem 1 in Multi-Armed Exponential Bandit [Gao et al]"
)
plt.xlabel("initial belief $p$")
plt.ylabel("value")
plt.show()
