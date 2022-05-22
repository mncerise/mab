import numpy as np
import matplotlib.pyplot as plt

from agent import Agent
from mab import MAB
import arm_settings as arm

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


def plot_gain(N, payoff_per_arm, cost_per_arm, rate_per_arm, info=False):
    p = np.linspace(0, 1, N + 1)
    # # offsets = p[:-1] * slopes - 0.5

    gain = np.max(
        payoff_per_arm * np.vstack(p) - cost_per_arm / rate_per_arm,
        axis=1,
        initial=0,
    )

    plt.plot(p, gain, label="projects")
    plt.ylim(-5, 60)

    if info:
        source_payoff = np.max(payoff_per_arm - cost_per_arm / rate_per_arm)
        source_gain = source_payoff * p - (source_payoff * 0.2)
        plt.plot(p, source_gain, label="source")

        source_gain = source_payoff * p - 29
        plt.plot(p, source_gain, label="source")
        plt.legend()

    plt.title(
        "The expected gain of the original projects and the source in terms of belief."
    )
    plt.xlabel("belief $p$")
    plt.ylabel(
        "expected gain $\max \{0,\ max_i \{\pi_i p - \dfrac{c_i}{\lambda_i}\}\}$"
    )
    plt.show()


plot_gain(N, payoff_per_arm, cost_per_arm, rate_per_arm)
