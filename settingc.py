import black
import numpy as np
import matplotlib.pyplot as plt

from agent import Agent
from mab import MAB
import arm_settings as arm

# MAB settings
N = 20

# Linear interpolation to approximate exponential growth
p = np.linspace(0, 1, N + 1)
vals = np.power(30, 1.5 * (p - 0.2)) - 1
slopes = (vals[1:] - vals[:-1]) / (p[1:] - p[:-1])
offsets = vals[:-1] - p[:-1] * slopes

# Set arm parameters
payoff_per_arm = slopes
cost_per_arm = -offsets
rate_per_arm = np.ones(N)


_, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 20))
# Plot the results of linear interpolation
for i in range(N):
    ax1.plot(p, slopes[i] * p + offsets[i])
ax1.plot(p, vals, "o", color="black")
ax1.set_ylabel("$\pi_i p - \dfrac{c_i}{\lambda_i}$ ($i \in \{1,\ldots, N\}$)")

# Plot the expected gain for each p
gain = np.max(
    payoff_per_arm * np.vstack(p) - cost_per_arm / rate_per_arm,
    axis=1,
    initial=0,
)
ax2.plot(p, gain, label="projects")
ax2.set_ylabel(
    "expected gain $\max \{0,\ max_i \{\pi_i p - \dfrac{c_i}{\lambda_i}\}\}$"
)

# Add labels, limits and title
for ax in (ax1, ax2):
    ax.set_xlabel("belief $p$")
    ax.set_ylim(-5, 60)

plt.suptitle(
    "The expected gain of the projects approximating "
    + "exponential growth through linear interpolation.",
)
plt.show()
