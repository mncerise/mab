"""
Author: Mara van der Meulen
---
Similar to first_strategy.py, but experiments are run once
with the optimal strategy, once with a random decision baseline
strategy and once with a minimal costs baseline strategy.
"""
import numpy as np
import matplotlib.pyplot as plt

from agent import Agent
from mab import MAB
import arm_settings as arm

# Set random seed for reproducibility
np.random.seed(56)

trials = 20000
p_vals = np.linspace(0, 1, 41)[:-1]
N = 20

# Choose mode "identical", "random", "specific"
arm_mode = "identical"
if arm_mode == "identical":
    payoff_per_arm, cost_per_arm, rate_per_arm = arm.setting_a()
elif arm_mode == "random":
    payoff_per_arm, cost_per_arm, rate_per_arm = arm.setting_b(N)
else:
    payoff_per_arm, cost_per_arm, rate_per_arm = arm.setting_c(N)


def plot_values(N, payoff_per_arm, cost_per_arm, rate_per_arm, label, base=0):
    """
    Plot the average eventual value with 95% confidence interval, the base
    parameter is used to indicate which strategy should be use.
    """
    mab = MAB(
        N,
        payoff_per_arm,
        cost_per_arm,
        rate_per_arm,
        info=False,
        b_ind=False,
    )

    agent = Agent(mab, 0.5)

    data = []
    for p in p_vals:
        agent.p0 = p
        agent.mab.priori = p
        agent.reset()

        vals = []
        success = 0
        for _ in range(trials):
            if base == 1:
                agent.baseline_strategy()
            elif base == 2:
                agent.baseline_strategy(random=False)
            else:
                agent.first_strategy()
            vals.append(agent.mab.value)

            if agent.mab.payoff != 0:
                success += 1

            agent.reset()
        data.append(vals)
        print(f"Successes: {success}/{trials}")

    avg_value = np.mean(data, axis=1)
    std_value = np.std(data, axis=1)
    plt.errorbar(
        p_vals,
        avg_value,
        yerr=2 * std_value / np.sqrt(trials),
        fmt=".",
        label=label,
    )


# Plot the results for all three strategies
_ = plt.figure(figsize=(8, 5))
plot_values(
    N,
    payoff_per_arm,
    cost_per_arm,
    rate_per_arm,
    "optimal strategy",
    base=0,
)
plot_values(
    N,
    payoff_per_arm,
    cost_per_arm,
    rate_per_arm,
    "baseline strategy (random)",
    base=1,
)
plot_values(
    N,
    payoff_per_arm,
    cost_per_arm,
    rate_per_arm,
    "baseline strategy (min costs)",
    base=2,
)

plt.xticks(p_vals[::4])

plt.title(
    "Agent's strategy based on Theorem 1 in Multi-Armed Exponential Bandit [Chen et al]"
)
plt.xlabel("initial belief $p$")
plt.ylabel("value")
plt.legend()
plt.show()
