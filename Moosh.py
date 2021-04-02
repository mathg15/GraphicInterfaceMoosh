import numpy as np


class Bragg:

    def __init__(self, periods, Lambda, Theta):
        self.Lambda = Lambda
        self.Theta = Theta
        self.Eps = np.array([1, 5, 2])
        self.Mu = np.array([1, 1, 1])

        self.n_periods = periods

        rep0 = np.tile([2, 3], (1, self.n_periods))
        rep0 = np.insert(rep0, 0, 1)
        rep0 = np.append(rep0,1)
        self.Type = rep0

        rep1 = np.tile([600 / np.sqrt(2), 600 / (4 * 2)], (1, self.n_periods))
        self.height = np.array([600, rep1, 100], dtype=object)

        self.pol = 1

    def cascade(self, A, B):
        t = 1 / (1 - B[1, 1] * A[2, 2])
        S = np.array([[A[1, 1] + A[1, 2] * B[1, 1] * A[2, 1] * t, A[1, 2] * B[1, 2] * t],
                      [B[2, 1] * A[2, 1] * t, B[2, 2] + A[2, 2] * B[1, 2] * B[2, 1] * t]])
        return S

    def coef(self, *args):

        if np.shape(args) == 0:
            self.pol = args[1]

        self.height[1] = 0

        if self.pol == 0:
            f = self.Mu
        else:
            f = self.Eps

        k0 = 2 * np.pi / self.Lambda
        g = len(self.Type)
        h = self.Type
        print(h)
        print(g)


a = Bragg(20, 600, (35 * np.pi) / 180)
a.coef()
