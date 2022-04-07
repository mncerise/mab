from agent import Agent
from mab import MAB

import numpy as np

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

agent = Agent(mab, 0.5)
agent.first_strategy()
