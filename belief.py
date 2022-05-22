import numpy as np
import matplotlib.pyplot as plt

from agent import Agent
from mab import MAB
import arm_settings as arm

trials = 200
p0 = 0.95

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


def plot_belief(dt, payoff_per_arm, cost_per_arm, rate_per_arm):
    mab = MAB(N, payoff_per_arm, cost_per_arm, rate_per_arm, b_ind=False)
    agent = Agent(mab, p0)

    # Set time step
    agent.mab.dt = dt

    # Compute belief using bayesian updates and adaptations
    p_bay = []
    p_upd = []
    for _ in range(trials):
        agent.first_strategy()
        p_bay.append(agent.p_bay)
        p_upd.append(agent.p_upd)
        agent.reset()

    # PLOT the longest single run
    # q = np.argmax([len(r) for r in p_bay])
    # p_bay_l = p_bay[q]
    # p_upd_l = p_upd[q]

    # plt.plot(
    #     np.arange(len(p_bay_l)) * agent.mab.dt,
    #     p_bay_l,
    #     label="Bayesian updates",
    # )
    # plt.plot(
    #     np.arange(len(p_upd_l)) * agent.mab.dt,
    #     p_upd_l,
    #     label="adaptations",
    # )

    # PLOT the maximal value over all runs (deterministic except for b=0 or 1)
    length = np.max([len(p) for p in p_bay])
    p_bay = np.array([np.pad(r, (0, length - len(r))) for r in p_bay])
    p_upd = np.array([np.pad(r, (0, length - len(r))) for r in p_upd])

    plt.plot(
        np.arange(np.shape(p_bay)[1]) * agent.mab.dt,
        np.max(p_bay, axis=0),
        label="Bayesian updates",
    )
    plt.plot(
        np.arange(np.shape(p_upd)[1]) * agent.mab.dt,
        np.max(p_upd, axis=0),
        label="adaptations",
    )

    # plt.errorbar(
    #     np.arange(np.shape(p_bay)[1]) * agent.mab.dt,
    #     np.mean(p_bay, axis=0),
    #     yerr=np.std(p_bay, axis=0),
    #     fmt=".",
    #     color="blue",
    #     capsize=2,
    # )
    # plt.errorbar(
    #     np.arange(np.shape(p_upd)[1]) * agent.mab.dt,
    #     np.mean(p_upd, axis=0),
    #     yerr=np.std(p_upd, axis=0),
    #     fmt=".",
    #     color="red",
    #     capsize=2,
    # )

    plt.xlim(0, 4.3)
    plt.ylim(0, 1)

    plt.title(f"$dt$ = {dt}")
    plt.xlabel("time $t$")
    plt.ylabel("belief $p$")
    plt.legend()


def belief_figure(payoff_per_arm, cost_per_arm, rate_per_arm):
    _, axes = plt.subplots(4, figsize=(15, 20))

    vals = [0.2, 0.1, 0.05, 0.01]

    # xlims = []
    for i in range(4):
        plt.subplot(2, 2, i + 1)
        plot_belief(vals[i], payoff_per_arm, cost_per_arm, rate_per_arm)
    # for i in range(len(axes)):
    # plt.subplot(2, 2, i + 1)
    # plot_belief(vals[i], payoff_per_arm, cost_per_arm, rate_per_arm)

    # print(xlims)
    # print(
    #     ax[0].get_xlim(),
    #     ax[1].get_xlim(),
    #     ax[2].get_xlim(),
    #     ax[3].get_xlim(),
    # )
    # print(ax[0])
    # plt.xlim(0, np.max(xlims))
    # ax[1].set_xlim(0, np.max(xlims))
    # ax[2].set_xlim(0, np.max(xlims))
    # ax[3].set_xlim(0, np.max(xlims))
    # plt.setp(ax, xlim=(0, np.max(xlims)))

    plt.suptitle(
        "Comparison of different methods to calculate the agent's belief over time."
    )

    plt.show()


belief_figure(payoff_per_arm, cost_per_arm, rate_per_arm)
