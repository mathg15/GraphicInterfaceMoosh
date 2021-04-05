import numpy as np


class Bragg:

    def __init__(self, periods, Lambda, Theta):
        self.Lambda = Lambda
        self.Theta = Theta
        self.Eps = np.array([1, 5, 2])
        self.Mu = np.array([1, 1, 1])

        self.n_periods = periods

        rep0 = np.tile([1, 2], (1, self.n_periods))
        rep0 = np.insert(rep0, 0, 0)
        rep0 = np.append(rep0, 0)
        self.Type = rep0

        rep1 = np.tile([600 / np.sqrt(2), 600 / (4 * 2)], (1, self.n_periods))
        rep1 = np.insert(rep1, 0, 600)
        rep1 = np.append(rep1, 100)
        self.height = rep1

        self.pol = 1

    def cascade(self, A, B):
        t = 1 / (1 - B[1, 1] * A[2, 2])
        S = np.array([[A[1, 1] + A[1, 2] * B[1, 1] * A[2, 1] * t, A[1, 2] * B[1, 2] * t],
                      [B[2, 1] * A[2, 1] * t, B[2, 2] + A[2, 2] * B[1, 2] * B[2, 1] * t]])
        return S

    def coef(self, *args):

        if np.shape(args) != 0:
            self.pol = args[0]

        self.height[0] = 0

        if self.pol == 0:
            f = self.Mu
        else:
            f = self.Eps

        k0 = (2 * np.pi) / self.Lambda
        print(k0)
        g = len(self.Type)

        TypeEps = np.copy(self.Type)
        TypeMu = np.copy(self.Type)

        for i in range(2, -1, -1):
            TypeEps[TypeEps == i] = self.Eps[i]
            TypeMu[TypeMu == i] = self.Mu[i]

        alpha = np.sqrt(TypeEps[0] * TypeMu[0]) * k0 * np.sin(self.Theta)
        gamma = np.sqrt(TypeEps * TypeMu * (k0 ** 2) - np.ones(g) * (alpha ** 2))

        if np.real(TypeEps[0]) < 0 and np.real(TypeMu[0]):
            gamma[0] = -gamma[0]

        if g > 2:
            gamma[1: g-1] = gamma[1:g-1] * (1-2*(np.imag(gamma[1:g-1]) < 0))

        if np.real(TypeEps[g]) < 0 and np.real(TypeMu[g]) < 0 and np.real(np.sqrt(TypeEps[g] * TypeMu * (k0**2) - (alpha**2))):
            gamma[g] = -np.sqrt(TypeEps[g] * TypeMu[g] - (alpha**2))
        else:
            gamma[g] = np.sqrt(TypeEps[g] * TypeMu[g] * (k0**2) - (alpha**2))






        print(gamma)
        print(self.Type)
        print(TypeEps)
        print(TypeMu)


theta = (35 * np.pi) / 180

a = Bragg(20, 600, theta)
a.coef()
