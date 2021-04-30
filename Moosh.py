import numpy as np
from scipy import special
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


class mat():

    def __init__(self, _lambda):

        self._lambda = _lambda
        self.i = complex(0, 1)

    def faddeva(self, z, n):

        try:
            len(arg) < 2
        except:
            n = np.array([])

        try:
            len(n) == 0
        except:
            n = 16

        w = np.zeros(z.shape, dtype=complex)

        idx = np.real(z) == 0
        w[idx] = np.exp(-z[idx] ** 2) * special.erfc(np.imag(z[idx]))

        idx = np.logical_not(idx)

        idx1 = idx & np.imag(z) < 0
        z[idx1] = np.conj(z[idx1])

        M = 2 * n
        M2 = M * 2
        k = np.conj(np.arange(-M + 1, M - 1, 1))
        L = np.sqrt(N / np.sqrt(2))

        theta = k * np.pi / M
        t = L * np.tan(theta / 2)
        f = np.exp(-t ** 2) * (L ** 2 + t ** 2)
        f = np.concatenate((0, f))
        a = np.real(np.fft.fft(np.fft.fftshift(f))) / M2
        a = np.flipud(a[np.arange(2, N + 1)])

        z = (L + self.i * z[idx]) / (L - self.i * z[idx])
        p = np.polyval(a, z)
        w[idx] = 2 * p / (L - self.i * z[idx]) ** 2 + (1 / np.sqrt(np.pi)) / (L - self.i * z[idx])

        w[idx1] = np.conj(2 * np.exp(-z[idx1]) ** 2 - w[idx1])

    def Aubb(self):

        w = 6.62606957e-25 * 299792458 / 1.602176565e-19 / self._lambda

        f0 = 0.770
        Gamma0 = 0.050
        omega_p = 9.03
        f = np.array([0.054, 0.050, 0.312, 0.719, 1.648])
        Gamma = np.array([0.074, 0.035, 0.083, 0.125, 0.179])
        omega = np.array([0.218, 2.885, 4.069, 6.137, 27.97])
        sigma = np.array([0.742, 0.349, 0.830, 1.246, 1.795])

        a = np.sqrt(w * (w + self.i * Gamma))
        a = a * np.sign(np.real(a))
        x = (a - omega) / (np.sqrt(2) * sigma)
        y = (a + omega) / (np.sqrt(2) * sigma)

        eps = 1 - omega_p ** 2 * f0 / (w * (w + self.i * Gamma0)) + np.sum(self.i * np.sqrt(np.pi) * f * omega_p ** 2 /
                                                                           (2 * np.sqrt(2) * a * sigma) * ())

    def h2o(self):

        a = np.array(
            [404.7, 435.8, 467.8, 480, 508.5, 546.1, 577, 579.1, 589.1, 643.8, 700, 750, 800, 850, 900, 950, 1000, 1050,
             1100])
        e = np.array([1.8056640625, 1.7988442641, 1.7932691569, 1.7914216336, 1.7875957401, 1.7833999936, 1.7804632356,
                      1.7802764329, 1.7794226025, 1.7753164081, 1.7718005881, 1.766241, 1.763584, 1.760929, 1.763584,
                      1.760929,
                      1.758276, 1.755625, 1.752976])
        eps = np.interp(self._lambda, a, ea)
        return eps


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
        rep1 = np.insert(rep1, 0, 1600)
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
        Re = np.abs(r) ** 2

        # Coefficient de transmission de l'énergie
        Tr = (np.abs(tr) ** 2) * gamma[g - 1] * f[0] / (gamma[0] * f[g - 1])

        return r, tr, Re, Tr

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

        print(self.height)

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

    def beam(self, _lambda, _theta, C):

        # Spatial window size
        d = 70 * _lambda

        # Incident beam width
        w = 10 * _lambda

        # Number of pixels horizontally
        nx = d / 20

        # Number of pixels verticaly
        ny = np.floor(self.height / 20)
        # print(ny)
        # Number of modes
        nmod = np.floor(0.83660 * (d / w))

        # print(nmod)

        self.height = self.height / d
        l = _lambda / d
        w = w / d

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

        k0 = (2 * np.pi) / l

        En = np.zeros((int(np.sum(ny)), int(nx)), dtype=complex)

        # g est la longeur de Type
        g = len(self.Type)

        # Amplitude of the different modes
        X = np.exp(-(w ** 2) * (np.pi ** 2) * np.arange(-nmod, nmod + 1) ** 2) * np.exp(
            2 * self.i * np.pi * np.arange(-nmod, nmod + 1) * C)

        # Scattering matrix
        T = np.zeros((2 * g, 2, 2), dtype=complex)
        T[0] = [[0, 1], [1, 0]]

        for nm in range(int(2 * nmod)):
            alpha = np.sqrt(TypeEps[0] * TypeMu[0]) * k0 * np.sin(_theta) + 2 * np.pi * (nm - nmod - 1)
            # print("alpha :",alpha)
            gamma = np.sqrt(TypeEps * TypeMu * (k0 ** 2) - np.ones(g) * (alpha ** 2))

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

            H = np.zeros((2 * g - 1, 2, 2), dtype=complex)
            H[0] = T[2 * g - 1]

            for j in range(len(T) - 2):
                A[j + 1] = self.cascade(A[j], T[j + 1])
                H[j + 1] = self.cascade(T[len(T) - 1 - j], H[j])

            # I = np.zeros((len(T), 2, 2), dtype=complex)

            for j in range(len(T) - 2):
                I = np.array([[A[j][1, 0], A[j][1, 1] * H[j][1, 1] * H[len(T) - 2 - j][0, 1]],
                              [A[j][1, 0] * H[len(T) - 2 - j][0, 0],
                               H[len(T) - 2 - j][0, 1] / (1 - A[j][1, 1]) * H[len(T) - 2 - j][0, 0]]], dtype=complex)

            h = 0
            I[2 * g - 1] = np.zeros((2, 2), dtype=complex)

            # Computation of the field in the different layers for one mode (plane wave)
            t = 1
            E = np.zeros((int(sum(ny)), 1), dtype=complex)

            for k in range(g - 1):
                for m in range(int(ny[k])):
                    h = h + self.height[k] / (ny[k])
                    # print(h)
                    E[t, 0] = I[2 * k - 1][0, 0] * np.exp(self.i * gamma[k] * h) + I[2 * k][1, 0] * np.exp(
                        self.i * gamma[k] * (self.height[k] - h))
                    print(E)

                    t = t + 1
                h = 0

            E = E * np.exp(self.i * alpha * np.arange(0, nx) / nx)
            # print(E)
            En = En + X[nm] * E
            # print(En)

        V = np.abs(En)
        # print(V)
        V_ = V.max()
        # plt.imshow((V / V_)*120)
        # plt.show()


theta = (35 * np.pi) / 180

a = Bragg(20, 200)

# a.angular(600)
# a.spectrum(theta)
a.beam(600, theta, 0.4)
