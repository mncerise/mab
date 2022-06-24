"""
Author: Mara van der Meulen
---
Similar to first_strategy.py, but experiments are run once with a fixed a
priori probability (resulting in a misspecification), and once with the a
priori probability of succes being set to match the initial belief. The a
priori probability refers to the probability that success is possible.
"""
import numpy as np
import matplotlib.pyplot as plt

from agent import Agent
from mab import MAB
import arm_settings as arm

# Set random seed for reproducibility
np.random.seed(56)

trials = 1000
p_vals = np.linspace(0, 1, 41)[:-1]
N = 20

# Choose mode "identical", "random", "specific"
arm_mode = "specific"
if arm_mode == "identical":
    payoff_per_arm, cost_per_arm, rate_per_arm = arm.setting_a()
elif arm_mode == "random":
    payoff_per_arm, cost_per_arm, rate_per_arm = arm.setting_b(N)
else:
    payoff_per_arm, cost_per_arm, rate_per_arm = arm.setting_c(N)


def plot_values(
    N, payoff_per_arm, cost_per_arm, rate_per_arm, label, priori=False
):
    """
    Plot the average eventual value of an agent using the optimal strategy
    when the MAB model is set according to the specified parameters. The
    priori parameter specifies whether the a priori probability should match
    the initial belief.
    """
    mab = MAB(
        N,
        payoff_per_arm,
        cost_per_arm,
        rate_per_arm,
        b_ind=False,
    )

    agent = Agent(mab, 0.5)

    # Run experiment
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
    plt.errorbar(
        p_vals,
        avg_value,
        yerr=2 * std_value / np.sqrt(trials),
        fmt=".",
        label=label,
    )


# Plot the results
_ = plt.figure(figsize=(8, 5))

plot_values(
    N,
    payoff_per_arm,
    cost_per_arm,
    rate_per_arm,
    "misspecification",
    priori=False,
)
plot_values(
    N,
    payoff_per_arm,
    cost_per_arm,
    rate_per_arm,
    "priori matches initial belief",
    priori=True,
)

plt.xticks(p_vals[::4])

plt.title(
    "Agent's strategy based on Theorem 1 in Multi-Armed Exponential Bandit [Chen et al]"
)
plt.xlabel("initial belief $p$")
plt.ylabel("value")
plt.legend()
plt.show()
