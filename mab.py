import numpy as np


class MAB:
    def __init__(self, N, payoff, cost, rate):
        self.N = N

        self.arm_payoff = np.resize(payoff, N)
        self.arm_cost = np.resize(cost, N)
        self.arm_rate = np.resize(rate, N)
        self.arm_success = np.random.randint(2, size=N)

        self.x = np.zeros(N)

        self.costs = 0
        self.payoff = 0
        self.value = 0

        self.dt = 0.001

    def select_arm(self, i):
        """
        Select arm i.
        """
        self.x = np.zeros(self.N)
        self.x[i] = 1

    def reset(self):
        """
        Resets the multi-armed bandit game.
        """
        self.arm_success = np.random.randint(2, size=self.N)
        self.x = np.zeros(self.N)

        self.costs = 0
        self.payoff = 0
        self.value = 0

    def timestep(self):
        """
        Simulate the effects of a single timestep.
        Each timestep has corresponding costs. If a breakthrough
        occurs, calculate payoff and update model.
        Returns: 1 if a breakthrough occurs, 0 otherwise.
        """
        self.costs += np.sum(self.x * self.arm_cost * self.dt)

        # Probability of breakthrough explained in Section 3.2
        possible = np.sum(self.arm_success * self.x)
        if np.random.uniform() < possible * (
            1 - np.exp(-np.sum(self.arm_rate * self.x) * self.dt)
        ):
            self.payoff += np.sum(self.arm_payoff * self.x)
            return 1

        return 0

    def quit(self):
        if self.payoff == 0:
            print("QUITTING WITHOUT BREAKTHROUGH")
        else:
            print("BREAKTHROUGH OCCURED")

        self.value = self.payoff - self.costs
        print("Value: %.2f" % self.value)

        return

    def play(self):
        """
        Interactive game between the agent (user) and the ARMs.
        ARMs are selected through user input.
        """
        while True:
            i = input("Choose an arm: ")

            if not i.isdigit() or not 0 <= int(i) < self.N:
                continue

            # Select ARM i during the next timestep
            self.select_arm(int(i))

            # Check for breakthrough
            if self.timestep() == 1:
                self.value = self.payoff - self.costs
                print("BREAKTHROUGH, value: %.2f" % self.value)

                return self.value

            print(
                "Total costs: %.2f" % self.costs, "Total payoff:", self.payoff
            )
