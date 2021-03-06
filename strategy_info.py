"""
Author: Mara van der Meulen
---
Similar to first_strategy.py, but experiments are run
once with and once without information source to compare,
or once with possibility of succes general and once with
possibility of success individual per arm.
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
arm_mode = "specific"
if arm_mode == "identical":
    payoff_per_arm, cost_per_arm, rate_per_arm = arm.setting_a()
elif arm_mode == "random":
    payoff_per_arm, cost_per_arm, rate_per_arm = arm.setting_b(N)
else:
    payoff_per_arm, cost_per_arm, rate_per_arm = arm.setting_c(N)

# Choose experiment "info" or "b_ind"
experiment = "info"


def plot_values(
    N,
    payoff_per_arm,
    cost_per_arm,
    rate_per_arm,
    label,
    info=False,
    b_ind=False,
):
    """
    Plot the average eventual value where the MAB model is set with the
    specified parameters and settings.
    """
    mab = MAB(
        N,
        payoff_per_arm,
        cost_per_arm,
        rate_per_arm,
        info=info,
        source=(np.max(payoff_per_arm - cost_per_arm / rate_per_arm) * 0.2, 1),
        b_ind=b_ind,
    )

    agent = Agent(mab, 0.5)

    # Run experiment
    data = []
    for p in p_vals:
        agent.p0 = p
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
if experiment == "info":
    plot_values(
        N,
        payoff_per_arm,
        cost_per_arm,
        rate_per_arm,
        "without information",
        info=False,
    )
    plot_values(
        N,
        payoff_per_arm,
        cost_per_arm,
        rate_per_arm,
        "information source",
        info=True,
    )
else:
    plot_values(
        N,
        payoff_per_arm,
        cost_per_arm,
        rate_per_arm,
        "general possibility",
        b_ind=False,
    )
    plot_values(
        N,
        payoff_per_arm,
        cost_per_arm,
        rate_per_arm,
        "possibility per arm",
        b_ind=True,
    )

plt.xticks(p_vals[::4])

plt.title(
    "Agent's strategy based on Theorem 1 in Multi-Armed Exponential Bandit [Chen et al]"
)
plt.xlabel("initial belief $p$")
plt.ylabel("value")
plt.legend()
plt.show()
