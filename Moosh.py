import numpy as np

class Bragg:


    def __init__(self, periods):
        self.Eps = np.array([1,5,2])
        self.Mu = np.array([1,1,1])

        self.n_periods = periods

        rep0 = np.tile([2, 3], (1, self.n_periods))
        self.Type = np.array([1, rep0, 1], dtype=object)

        rep1 = np.tile([600 / np.sqrt(2), 600 / (4 * 2)], (1, self.n_periods))
        self.height = np.array([600, rep1, 100], dtype=object)

        self.pol = 1

    def cascade(self,A,B):
        t = 1 / (1-B[1, 1]*A[2, 2])
        self.S = np.array([[A[1, 1] + A[1, 2]*B[1, 1]*A[2, 1]*t, A[1, 2]*B[1, 2]*t], [B[2, 1]*A[2, 1]*t, B[2, 2]+A[2, 2]*B[1, 2]*B[2, 1]*t]])

    def coef(self):

