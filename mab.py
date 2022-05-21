import numpy as np


class MAB:
    def __init__(
        self,
        N,
        payoff,
        cost,
        rate,
        priori=0.5,
        info=False,
        source=(0, 0),
        b_ind=False,
        log=False,
    ):
        self.N = N

        self.arm_payoff = np.resize(payoff, N)
        self.arm_cost = np.resize(cost, N)
        self.arm_rate = np.resize(rate, N)

        self.costs = 0
        self.payoff = 0
        self.value = 0

        self.dt = 0.01
        self.log = log

        # If set, an information source is added
        if info:
            self.N = N + 1

            self.arm_payoff = np.append(
                self.arm_payoff,
                np.max(self.arm_payoff - self.arm_cost / self.arm_rate),
            )
            self.arm_cost = np.append(self.arm_cost, source[0])
            self.arm_rate = np.append(self.arm_rate, source[1])

        self.x = np.zeros(self.N)

        self.priori = priori
        # If set, possibility of succes is arm dependent
        self.b_ind = b_ind
        if b_ind:
            # self.arm_success = np.random.randint(2, size=self.N)
            self.arm_success = np.random.binomial(1, priori, self.N)
        else:
            # self.arm_success = np.resize(np.random.randint(2), self.N)
            self.arm_success = np.resize(
                np.random.binomial(1, priori, 1), self.N
            )

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
        if self.b_ind:
            # self.arm_success = np.random.randint(2, size=self.N)
            self.arm_success = np.random.binomial(1, self.priori, self.N)
        else:
            # self.arm_success = np.resize(np.random.randint(2), self.N)
            self.arm_success = np.resize(
                np.random.binomial(1, self.priori, 1), self.N
            )

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
        """
        The agent quits, this can be interpreted equivalently with
        setting resource allocation to zero for all arms.
        """
        if self.log and self.payoff == 0:
            print("QUITTING WITHOUT BREAKTHROUGH")
        elif self.log:
            print("BREAKTHROUGH OCCURED")

        self.value = self.payoff - self.costs
        # print("Value: %.2f" % self.value)

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
