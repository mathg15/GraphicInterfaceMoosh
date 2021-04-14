import numpy as np


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
        t = 1 / (1 - B[0, 1] * A[1, 1])
        S = np.array([[A[0, 0] + A[0, 1] * B[0, 0] * A[1, 0] * t, A[0, 1] * B[0, 1] * t],
                      [B[1, 0] * A[1, 0] * t, B[1, 1] + A[1, 1] * B[0, 1] * B[1, 0] * t]])
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
            gamma[1: g - 2] = gamma[1:g - 2] * (1 - 2 * (np.imag(gamma[1:g - 2]) < 0))
        # Condition de l'onde sortante pour le dernier milieu
        if np.real(TypeEps[g - 1]) < 0 and np.real(TypeMu[g - 1]) < 0 and np.real(
                np.sqrt(TypeEps[g - 1] * TypeMu * (k0 ** 2) - (alpha ** 2))):
            gamma[g - 1] = -np.sqrt(TypeEps[g - 1] * TypeMu[g - 1] - (alpha ** 2))
        else:
            gamma[g - 1] = np.sqrt(TypeEps[g - 1] * TypeMu[g - 1] * (k0 ** 2) - (alpha ** 2))

        # Définition de la matrice T
        # T = np.array([[[0, 1], [1, 0]], [[0, 0], [0, 0]], [[0, 0], [0, 0]]])
        # # Cacul des matrices S
        # for k in range(0, g - 2, 1):
        #     # Matrice de diffusion des couches
        #     t = np.exp(1j * gamma[k] * self.height[k])
        #     T[2 * k, :, :] = np.array([[0, t], [t, 0]])
        #
        #     # Matrice de diffusion d'interface
        #     b1 = gamma[k] / (f[k])
        #     b2 = gamma[k + 1] / (f[k + 1])
        #     T[2 * k + 1, :, :] = np.array(
        #         [[(b1 - b2) / (b1 + b2), (2 * b2) / (b1 + b2)], [(2 * b1) / (b1 + b2), (b2 - b1) / (b1 + b2)]])
        #
        # # Matrice de diffusion pour la dernière couche
        # t = np.exp(1j * gamma[g - 1] * self.height[g - 1])
        # T[2 * g, :, :] = np.array([[0, t], [t, 0]])

        a = np.array([[[1, 0], [0, 1]]])

        for k in range(0, g-1, 1):
            t = np.exp(1j * gamma[k] * self.height[k])
            B = np.array([[[t, 0], [0, t]]])

            b1 = gamma[k] / (f[k])
            b2 = gamma[k + 1] / (f[k + 1])
            # Ajouter des crochets
            C = np.array(
                [[[(b1 - b2) / (b1 + b2), (2 * b2) / (b1 + b2)], [(2 * b1) / (b1 + b2), (b2 - b1) / (b1 + b2)]]])
            if k % 2 == 0:
                a = np.concatenate((a, B), axis=0)
            else:
                a = np.concatenate((a, C), axis=0)

        D = np.array([[[t, 0], [0, t]]])
        a = np.concatenate((a, D), axis=0)

        print(a)

        # On combine les différentes matrices
        # A = np.array([[[0, 0], [0, 0]], [[0, 0], [0, 0]], [[0, 0], [0, 0]]])
        # A[0, :, :] = T[0, :, :]
        # for i in range(0, T.shape[1] - 2, 1):
        #     A[i + 1] = self.cascade(A[i, :, :], T[i + 1, :, :])

        # Coefficient de reflexion de l'ensemble de la structure
        r = a[a.shape[1], :, :][0, 0]
        print(r)
        # Coefficient de transmission de l'ensemble de la structure
        tr = a[a.shape[1], :, :][1, 0]
        print(tr)
        # Coefficient de réflexion de l'énergie
        R = np.abs(r) ** 2
        print(R)
        # Coefficient de transmission de l'énergie
        Tr = (np.abs(t) ** 2) * gamma[g - 1] * f[0] / (gamma[0] * f[g - 1])
        print(Tr)

        return r, tr, R, Tr


theta = (35 * np.pi) / 180

a = Bragg(20, 600, theta)
a.coefficient()



