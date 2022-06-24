"""
Author: Mara van der Meulen
---
Plot the average eventual value with 95% confidence interval for
an agent using the optimal strategy. This strategy corresponds to
Theorem 1 in the paper Multi-Armed Exponential Bandits by Chen et al.
"""
import numpy as np
import matplotlib.pyplot as plt

from agent import Agent
from mab import MAB
import arm_settings as arm

# Set random seed for reproducibility
np.random.seed(56)

# Number of iterations
trials = 1000
p_vals = np.linspace(0, 1, 41)[:-1]

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

mab = MAB(
    N,
    payoff_per_arm,
    cost_per_arm,
    rate_per_arm,
    info=False,
    source=(np.max(payoff_per_arm - cost_per_arm / rate_per_arm) * 0.2, 1),
    b_ind=False,
)
agent = Agent(mab, 0.5)

# Run experiment
data = []
for p in p_vals:
    agent.p0 = p
    agent.reset()

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

# Plot the results
_ = plt.figure(figsize=(8, 5))

avg_value = np.mean(data, axis=1)
std_value = np.std(data, axis=1)
plt.errorbar(
    p_vals,
    avg_value,
    yerr=2 * std_value / np.sqrt(trials),
    fmt=".",
    color="red",
)

plt.xticks(p_vals[::4])

plt.title(
    "Agent's strategy based on Theorem 1 in Multi-Armed Exponential Bandit [Chen et al]"
)
plt.xlabel("initial belief $p$")
plt.ylabel("value")
plt.show()
