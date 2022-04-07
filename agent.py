import numpy as np
from soupsieve import select


class Agent:
    def __init__(self, mab, p):
        self.mab = mab
        self.p0 = p
        self.p = p

        self.depth = 0

    def update_belief(self):
        """
        Update the agent's belief over a single time interval of
        size dt, assuming. See Equation 3.10 in Section 3.3.
        """
        self.p = (
            self.p
            + self.p
            * (1 - self.p)
            * np.sum(self.mab.arm_rate * self.mab.x)
            * self.mab.dt
        )

    def reset(self):
        """
        Resets the agent belief to the initial belief, and
        resets the corresponding MAB game
        """
        self.p = self.p0

        self.mab.reset()

    def V(self, dt):
        breakthrough = self.p * np.sum(
            self.mab.arm_rate * self.x * self.mab.arm_payoff * dt
        )
        flowcost = dt * np.sum(self.x * self.arm_cost)
        future = 1 - self.p * np.sum(self.mab.arm_rate * self.x * dt) * self.V(
            dt
        )

        return breakthrough - flowcost + future

    def first_strategy(self):
        """
        Optimal strategy based on the first theorem in
        Multi-Armed Exponential Bandit [Lang].
        """
        if self.depth > 4500 or self.p <= np.min(
            self.mab.arm_cost / (self.mab.arm_rate * self.mab.arm_payoff)
        ):
            self.mab.quit()
        else:
            # Choose random project i that maximizes (pi_i p - c_i / lambda_i)
            vals = (
                self.mab.arm_rate * self.p
                - self.mab.arm_cost / self.mab.arm_rate
            )
            i = np.random.choice(np.flatnonzero(vals == vals.max()))
            self.mab.select_arm(i)
            # print(i)

            self.depth += 1

            # Quit when breakthrough happens
            if self.mab.timestep() == 1:
                self.mab.quit()
            else:
                self.first_strategy()

    def basic_strategy(self):
        self.mab.play()
        while self.mab.timestep() == 0:
            for i in range(self.mab.N):
                print(i)
                self.mab.select_arm(i)
        print("BREAKTHROUGH")
