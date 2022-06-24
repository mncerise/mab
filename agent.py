"""
Author: Mara van der Meulen
---
This file contains a class representing an agent that is
assigned a Multi-Armed Bandit model to interact with.
"""
import numpy as np


class Agent:
    def __init__(self, mab, p):
        self.mab = mab
        self.p0 = p
        self.p = p

        self.depth = 0

        # Bayesian updates
        self.exponent = 0
        self.p_vals = [self.p0]

        self.p_upd = [self.p0]
        self.p_bay = [self.p0]

    def bay_updates(self):
        """
        Update the agent's belief over a single time interval of size dt
        using Bayesian updates. See Equation 2.11 in Section 2.3.
        """
        self.exponent += np.sum(self.mab.arm_rate * self.mab.x) * self.mab.dt
        prob = self.p0 * np.exp(-self.exponent)
        self.p_bay.append(prob / (prob + (1 - self.p0)))

        self.p = self.p_bay[-1]

    def update_belief(self):
        """
        Update the agent's belief over a single time interval of size dt
        by adapting the previous belief. See Equation 2.12 in Section 2.3.
        """
        prev = self.p_upd[-1]
        self.p_upd.append(
            prev
            - (
                prev
                * (1 - prev)
                * np.sum(self.mab.arm_rate * self.mab.x)
                * self.mab.dt
            )
        )

    def reset(self):
        """
        Resets the agent belief to the initial belief, and
        resets the corresponding MAB model.
        """
        self.depth = 0

        self.p = self.p0
        self.p_vals = [self.p0]

        self.exponent = 0
        self.p_bay = [self.p0]
        self.p_upd = [self.p0]

        self.mab.reset()

    def first_strategy(self):
        """
        Optimal strategy based on Theorem 1 in the paper Multi-Armed
        Exponential Bandit by [Chen et al]. Choose the arm with maximal
        expected gain, unless expected gain is negative for all arms.
        """
        while self.p > np.min(
            self.mab.arm_cost / (self.mab.arm_rate * self.mab.arm_payoff)
        ):
            # Choose project i that maximizes (pi_i p - c_i / lambda_i)
            vals = (
                self.mab.arm_rate * self.p
                - self.mab.arm_cost / self.mab.arm_rate
            )
            i = np.argmax(vals)
            self.mab.select_arm(i)

            self.update_belief()
            self.bay_updates()
            # self.p_vals.append(self.p)

            if self.mab.timestep():
                break

        self.mab.quit()

    def baseline_strategy(self, random=True):
        """
        Baseline strategy where arms are chosen randomly. After each time
        step, there is a 20% probability of quitting. Alternatively, a minimal
        costs baseline strategy with 1% quitting probability can be used.
        """
        while np.random.uniform() > 0.2 * random + (0.01) * (not random):
            if random:
                i = np.random.randint(self.mab.N)
            else:
                i = np.argmin(self.mab.arm_cost)

            self.mab.select_arm(i)

            if self.mab.timestep() == 1:
                break
        self.mab.quit()
