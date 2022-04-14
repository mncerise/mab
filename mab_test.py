from agent import Agent
from mab import MAB

import numpy as np
import matplotlib.pyplot as plt

N = 10

payoff_per_arm = 1.5
cost_per_arm = 1
rate_per_arm = 1.5

ARMs_payoff = payoff_per_arm * np.ones(N)
ARMs_cost = cost_per_arm * np.ones(N)
ARMs_rate = rate_per_arm * np.ones(N)


mab = MAB(N, payoff_per_arm, cost_per_arm, rate_per_arm)
mab = MAB(N, ARMs_payoff, ARMs_cost, ARMs_rate)

# mab.play()

agent = Agent(mab, 0.9)
# agent.first_strategy()


def first_strategy(agent, p_vals):
    """
    Optimal strategy based on the first theorem in
    Multi-Armed Exponential Bandit [Lang].
    """
    p_vals.append(agent.p)
    if agent.depth > 4500 or agent.p <= np.min(
        agent.mab.arm_cost / (agent.mab.arm_rate * agent.mab.arm_payoff)
    ):
        agent.mab.quit()
        return p_vals
    else:
        # Choose random project i that maximizes (pi_i p - c_i / lambda_i)
        vals = (
            agent.mab.arm_rate * agent.p
            - agent.mab.arm_cost / agent.mab.arm_rate
        )
        i = np.random.choice(np.flatnonzero(vals == vals.max()))
        agent.mab.select_arm(i)
        agent.update_belief()
        print(i)

        agent.depth += 1

        # Quit when breakthrough happens
        if agent.mab.timestep() == 1:
            agent.mab.quit()
            return p_vals
        else:
            return first_strategy(agent, p_vals)


p_vals = []
p_vals = first_strategy(agent, p_vals)

plt.plot(range(len(p_vals)), p_vals)

plt.show()
