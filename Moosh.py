import numpy as np
import matplotlib.pyplot as plt


class Bragg:

    def __init__(self, periods, Npoints):

        # On définie la permittivité et la perméabilité
        self.Eps = np.array([1, 2, 4])
        self.Mu = np.array([1, 1, 1])

        # On définie le nombre de période
        self.n_periods = periods

        # On définie le nombre de points

        self.Npoints = Npoints

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

        self.i = complex(0, 1)

    def cascade(self, A, B):
        # On combine deux matrices de diffusion A et B en une seule matrice de diffusion (2*2) S
        t = 1 / (1 - B[0, 0] * A[1, 1])
        S = np.array([[A[0, 0] + A[0, 1] * B[0, 0] * A[1, 0] * t, A[0, 1] * B[0, 1] * t],
                      [B[1, 0] * A[1, 0] * t, B[1, 1] + A[1, 1] * B[0, 1] * B[1, 0] * t]], dtype=complex)

        return S

    def coefficient(self, thetacoef, _lambda):

        # On considère que la première valeur de la hauteur est 0
        self.height[0] = 0

        # On definie k0 à partir de lambda
        k0 = (2 * np.pi) / _lambda

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
        alpha = np.sqrt(TypeEps[0] * TypeMu[0]) * k0 * np.sin(thetacoef * (np.pi / 180))
        gamma = np.sqrt(TypeEps * TypeMu * (k0 ** 2) - np.ones(g) * (alpha ** 2))
        # print("gamma=",gamma)

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
        T = np.zeros((2 * g, 2, 2), dtype=complex)
        T[0] = [[0, 1], [1, 0]]
        # T = np.array([[[0, 1], [1, 0]], [[0, 0], [0, 0]], [[0, 0], [0, 0]]])
        # Cacul des matrices S
        for k in range(g - 1):
            # Matrice de diffusion des couches
            t = np.exp(self.i * gamma[k] * self.height[k])
            T[2 * k + 1] = np.array([[0, t], [t, 0]])

            # Matrice de diffusion d'interface
            b1 = gamma[k] / (f[k])
            b2 = gamma[k + 1] / (f[k + 1])
            T[2 * k + 2] = [[(b1 - b2) / (b1 + b2), (2 * b2) / (b1 + b2)],
                            [(2 * b1) / (b1 + b2), (b2 - b1) / (b1 + b2)]]

        # Matrice de diffusion pour la dernière couche
        t = np.exp(self.i * gamma[g - 1] * self.height[g - 1])
        T[2 * g - 1] = [[0, t], [t, 0]]

        A = np.zeros((2 * g - 1, 2, 2), dtype=complex)
        A[0] = T[0]

        for j in range(len(T) - 2):
            A[j + 1] = self.cascade(A[j], T[j + 1])

        # Coefficient de reflexion de l'ensemble de la structure
        r = A[len(A) - 1][0, 0]

        # Coefficient de transmission de l'ensemble de la structure
        tr = A[len(A) - 1][1, 0]

        # Coefficient de réflexion de l'énergie
        R = np.abs(r) ** 2

        # Coefficient de transmission de l'énergie
        Tr = (np.abs(tr) ** 2) * gamma[g - 1] * f[0] / (gamma[0] * f[g - 1])

        return r, tr, R, Tr

    def affichageCoef(self, angle, longueurOnde):

        reflexionT, transmissionT, reflexionE, transmissionE = self.coefficient(angle, longueurOnde)

        print("Coefficient de reflexion de l'ensemble de la structure :", reflexionT)
        print("Coefficient de transmission de l'ensemble de la structure :", transmissionT)
        print("Coefficient de transmission de l'énergie :", reflexionE)
        print("Coefficient de transmission de l'énergie :", transmissionE)

    def angular(self, lambda_):

        # Intervalle angulaire
        maxAngle = 89
        minAngle = 0
        rangeAngle = np.linspace(minAngle, maxAngle, self.Npoints)

        # Création des matrices
        a = np.ones((self.Npoints, 1), dtype=complex)
        b = np.ones((self.Npoints, 1), dtype=complex)
        c = np.ones((self.Npoints, 1), dtype=complex)
        d = np.ones((self.Npoints, 1), dtype=complex)

        for i in range(self.Npoints):
            tht = rangeAngle[i]
            a[i], b[i], c[i], d[i] = self.coefficient(tht, lambda_)

        plt.figure(1)
        plt.subplot(211)
        plt.title("Reflexion for lambda = 600 nm")
        plt.plot(rangeAngle, abs(c))
        plt.ylabel("Reflexion")
        plt.xlabel("Angle (degrees)")
        plt.subplot(212)
        plt.plot(rangeAngle, np.angle(a))
        plt.ylabel("Phase")
        plt.xlabel("Angle")
        plt.title("Phase of the reflection coefficient for lambda = 600 nm")
        plt.tight_layout()
        plt.show()

    def spectrum(self, theta_):

        # Intervalle de longeur d'onde
        minLambda = 400
        maxLambda = 700
        rangeLambda = np.linspace(minLambda, maxLambda, self.Npoints)

        a = np.ones((self.Npoints, 1), dtype=complex)
        b = np.ones((self.Npoints, 1), dtype=complex)
        c = np.ones((self.Npoints, 1), dtype=complex)
        d = np.ones((self.Npoints, 1), dtype=complex)

        for i in range(self.Npoints):
            rL = rangeLambda[i]
            a[i], b[i], c[i], d[i] = self.coefficient(theta_, rL)

        plt.figure(1)
        plt.subplot(211)
        plt.title("Reflexion for lambda = 600 nm")
        plt.plot(rangeLambda, abs(c))
        plt.ylabel("Reflexion")
        plt.xlabel("Angle (degrees)")
        plt.subplot(212)
        plt.plot(rangeLambda, np.angle(a))
        plt.ylabel("Phase")
        plt.xlabel("Angle")
        plt.title("Phase of the reflection coefficient for lambda = 600 nm")
        plt.tight_layout()
        plt.show()


theta = (35 * np.pi) / 180

a = Bragg(20, 200)

a.angular(600)
a.spectrum(theta)


