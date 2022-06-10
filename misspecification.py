# Similar to first_strategy.py, but experiments
# are run once with and once without
# information source to compare
import numpy as np
import matplotlib.pyplot as plt

from agent import Agent
from mab import MAB
import arm_settings as arm

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


_ = plt.figure(figsize=(10, 6))

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

plt.xticks(p_vals[::2])

plt.title(
    "Agent's strategy based on Theorem 1 in Multi-Armed Exponential Bandit [Gao et al]"
)
plt.xlabel("initial belief $p$")
plt.ylabel("value")
plt.legend()
plt.show()
