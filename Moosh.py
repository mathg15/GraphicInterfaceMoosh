import numpy as np
from simpy import I


class Bragg:

    def __init__(self, periods, Lambda, Theta):

        # On prend des valeurs de Lambda et Theta
        self.Lambda = Lambda
        self.Theta = Theta

        # On définie la permittivité et la perméabilité
        self.Eps = np.array([1, 2, 4])
        self.Mu = np.array([1, 1, 1])

        # On définie le nombre de période
        self.n_periods = periods

        # Définition de la structure du matériau => ( 0, 1, 2, 1, 2, ...., 1, 2, 0)
        rep0 = np.tile([1, 2], (1, self.n_periods))
        rep0 = np.insert(rep0, 0, 0)
        rep0 = np.append(rep0, 0)
        self.Type = rep0

        # Idem pour la hauteur
        rep1 = np.tile([600 / np.sqrt(2), 600 / (4 * 2)], (1, self.n_periods))
        rep1 = np.insert(rep1, 0, 600)
        rep1 = np.append(rep1, 100)
        self.height = rep1

        # Définition de la polarisation
        self.pol = 1

    def cascade(self, A, B):
        # On combine deux matrices de diffusion A et B en une seule matrice de diffusion (2*2) S
        t = 1 / (1 - B[1, 1] * A[2, 2])
        S = np.array([[A[1, 1] + A[1, 2] * B[1, 1] * A[2, 1] * t, A[1, 2] * B[1, 2] * t],
                      [B[2, 1] * A[2, 1] * t, B[2, 2] + A[2, 2] * B[1, 2] * B[2, 1] * t]])
        return S

    def coefficient(self):

        # On considère que la première valeur de la hauteur est 0
        self.height[0] = 0

        # On definie k0 à partir de lambda
        k0 = (2 * np.pi) / self.Lambda

        # g est la longeur de Type
        g = len(self.Type)

        # On définie deux array comme des copies de Type
        TypeEps = np.copy(self.Type)
        TypeMu = np.copy(self.Type)

        # On implémente les valeurs de la permittivité et de la perméabilité dans type pour définir le matériau
        for i in range(2, -1, -1):
            TypeEps[TypeEps == i] = self.Eps[i]
            TypeMu[TypeMu == i] = self.Mu[i]

        # En fonction de la polarisation, f prend soit la valeur TypeMu ou la valeur TypeEps
        if self.pol == 0:
            f = TypeMu
        else:
            f = TypeEps

        # Définition de alpha et gamma en fonction de TypeEps, TypeMu, k0 et Theta
        alpha = np.sqrt(TypeEps[0] * TypeMu[0]) * k0 * np.sin(self.Theta)
        gamma = np.sqrt(TypeEps * TypeMu * (k0 ** 2) - np.ones(g) * (alpha ** 2))

        # On fait en sorte d'avoir un résultat positif en foction au cas où l'index serait négatif
        if np.real(TypeEps[0]) < 0 and np.real(TypeMu[0]):
            gamma[0] = -gamma[0]

        # On modifie la détermination de la racine carrée pour obtenir un stabilité parfaite
        if g > 2:
            gamma[1: g - 1] = gamma[1:g - 1] * (1 - 2 * (np.imag(gamma[1:g - 1]) < 0))
        # Condition de l'onde sortante pour le dernier milieu
        if np.real(TypeEps[g]) < 0 and np.real(TypeMu[g]) < 0 and np.real(
                np.sqrt(TypeEps[g] * TypeMu * (k0 ** 2) - (alpha ** 2))):
            gamma[g] = -np.sqrt(TypeEps[g] * TypeMu[g] - (alpha ** 2))
        else:
            gamma[g] = np.sqrt(TypeEps[g] * TypeMu[g] * (k0 ** 2) - (alpha ** 2))

        # Définition de la matrice T
        T = np.ndarray([[0, 1], [1, 0]], [[0, 0], [0, 0]], [[0, 0], [0, 0]])

        # Cacul des matrices S
        for k in range(0, g - 1, 1):
            # Matrice de diffusion des couches
            t = np.exp(I * gamma(k) * hauteur(k))
            T[2 * k, :, :] = np.ndarray([[0, t], [t, 0]], [[0, 0], [0, 0]], [[0, 0], [0, 0]])

            # Matrice de diffusion d'interface
            b1 = gamma[k] / (f[k])
            b2 = gamma[k] / (f[k + 1])
            T[2 * k + 1, :, :] = np.ndarray(
                [[(b1 - b2) / (b1 + b2), (2 * b2) / (b1 + b2)], [(2 * b1) / (b1 + b2), (b2 - b1) / (b1 + b2)]],
                [[0, 0], [0, 0]], [[0, 0], [0, 0]])

        # Matrice de diffusion pour la dernière couche
        t = np.exp(I * gamma[g] * self.height[g])
        T[2 * g, :, :] = np.ndarray([[0, t], [t, 0]], [[0, 0], [0, 0]], [[0, 0], [0, 0]])

        # On combine les différentes matrices
        A[0, :, :] = T[0, :, :]
        for i in range(0, T.shape[1] - 2, 1):
            A[i + 1] = self.cascade(A[i, :, :], T[i + 1, :, :])

        # Coefficient de reflexion de l'ensemble de la structure
        r = A[A.shape[1], :, :][0, 0]
        # Coefficient de transmission de l'ensemble de la structure
        t = A[A.shape[1], :, :][1, 0]
        # Coefficient de réflexion de l'énergie
        R = np.abs(r) ** 2
        # Coefficient de transmission de l'énergie
        T = (np.abs(t) ** 2) * gamma[g] * f[0] / (gamma[1] * f[g])

        return r, t, R, T


theta = (35 * np.pi) / 180

a = Bragg(20, 600, theta)
a.coefficient()
