import numpy as np
import matplotlib.pyplot as plt

from agent import Agent
from mab import MAB
import arm_settings as arm

# Number of iterations
trials = 1000
p_vals = np.linspace(0, 1, 21)[:-1]

# MAB settings
N = 20

# Choose mode "identical", "random", "specific"
arm_mode = "identical"
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

_ = plt.figure(figsize=(15, 20))

avg_value = np.mean(data, axis=1)
std_value = np.std(data, axis=1)
plt.errorbar(p_vals, avg_value, yerr=std_value, fmt=".", color="red")

plt.xticks(p_vals[::2])

plt.title(
    "Agent's strategy based on Theorem 1 in Multi-Armed Exponential Bandit [Chen et al]"
)
plt.xlabel("initial belief $p$")
plt.ylabel("value")
plt.show()
