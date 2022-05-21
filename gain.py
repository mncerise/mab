import black
import numpy as np
import matplotlib.pyplot as plt

from agent import Agent
from mab import MAB

# MAB settings
N = 20
# random = True

payoff_per_arm = 20
cost_per_arm = 6
rate_per_arm = 3

# payoff_per_arm = np.linspace(0, 50, N)

# if random:
#     payoff_per_arm = np.random.uniform(0, 20, size=N)
#     cost_per_arm = np.random.uniform(5, 10, size=N)
#     rate_per_arm = np.random.uniform(0.01, 8, size=N)


def plot_gain(N, info=True):
    p = np.linspace(0, 1, N + 1)
    vals = np.power(30, 1.5 * (p - 0.2)) - 1
    slopes = (vals[1:] - vals[:-1]) / (p[1:] - p[:-1])
    offsets = vals[:-1] - p[:-1] * slopes
    # offsets = p[:-1] * slopes - 0.5

    # Set arm parameters
    payoff_per_arm = slopes
    cost_per_arm = -offsets
    rate_per_arm = np.ones(N)

    gain = np.max(
        payoff_per_arm * np.vstack(p) - cost_per_arm / rate_per_arm,
        axis=1,
        initial=0,
    )

    plt.plot(p, gain, label="projects")

    # for i in range(N):
    #     plt.plot(p, slopes[i] * p + offsets[i])
    # plt.plot(p, vals, "o", color="black")
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


plot_gain(N)
