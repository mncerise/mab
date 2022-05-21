# Similar to first_strategy.py, but experiments
# are run once with and once without
# information source to compare
import numpy as np
import matplotlib.pyplot as plt

from agent import Agent
from mab import MAB
import arm_settings as arm

trials = 100
p_vals = np.linspace(0, 1, 41)[:-1]
N = 10

# Choose mode "identical", "random", "specific"
arm_mode = "specific"
if arm_mode == "identical":
    payoff_per_arm, cost_per_arm, rate_per_arm = arm.setting_a()
elif arm_mode == "random":
    payoff_per_arm, cost_per_arm, rate_per_arm = arm.setting_b(N)
else:
    payoff_per_arm, cost_per_arm, rate_per_arm = arm.setting_c(N)


def plot_values(label, mode=0):
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
        if mode == 1 or mode == 2:
            agent.mab.priori = p
        if mode == 0 or mode == 2:
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
    plt.errorbar(p_vals, avg_value, yerr=std_value, fmt=".", label=label)


plot_values("misspecification, priori fixed", mode=0)
plot_values("misspecification, initial belief fixed", mode=1)
plot_values("priori matches initial belief", mode=2)

plt.xticks(p_vals[::2])

plt.title(
    "Agent's strategy based on Theorem 1 in Multi-Armed Exponential Bandit [Gao et al]"
)
plt.xlabel("initial belief / priori probability $\Pr[b=1]$")
plt.ylabel("value")
plt.legend()
plt.show()
