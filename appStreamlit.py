import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import matplotlib.image as im


# Bragg
class Bragg:

    def __init__(self, periods, Npoints, pol):

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
        self.pol = pol

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

        st.write("Coefficient de reflexion de l'ensemble de la structure :", reflexionT)
        st.write("Coefficient de transmission de l'ensemble de la structure :", transmissionT)
        st.write("Coefficient de transmission de l'énergie :", reflexionE)
        st.write("Coefficient de transmission de l'énergie :", transmissionE)

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
        plt.title("Phase of the Reflexion coefficient")
        plt.tight_layout()
        st.pyplot(plt)

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
        plt.title("Reflexion for lambda")
        plt.plot(rangeLambda, abs(c))
        plt.ylabel("Reflexion")
        plt.xlabel("Wavelength")
        plt.subplot(212)
        plt.plot(rangeLambda, np.angle(a))
        plt.ylabel("Phase")
        plt.xlabel("Wavelength")
        plt.title("Phase of the Reflexion coefficient")
        plt.tight_layout()
        st.pyplot(plt)

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
        g = len(self.Type - 1)

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
                H[j + 1] = self.cascade(T[len(T) - 2 - j], H[j])

            I = np.zeros((len(T), 2, 2), dtype=complex)

            for j in range(len(T) - 1):
                I = np.array([[A[j][1, 0], A[j][1, 1] * H[j][1, 1] * H[len(T) - 2 - j][0, 1]],
                              [A[j][1, 0] * H[len(T) - 2 - j][0, 0],
                               H[len(T) - 2 - j][0, 1] / (1 - A[j][1, 1]) * H[len(T) - 2 - j][0, 0]]], dtype=complex)

            h = 0
            # I[2 * g - 1] = np.zeros((1), dtype=complex)
            # print(I)
            # Computation of the field in the different layers for one mode (plane wave)
            t = 0
            E = np.zeros((int(sum(ny)), 2, 2), dtype=complex)
            # print(E)

            for k in range(g - 1):
                for m in range(int(ny[k])):
                    h = h + self.height[k] / ny[k]
                    # print(h)
                    E[t, 0] = I[2 * k - 2][0, 0] * np.exp(self.i * gamma[k] * h) + I[2 * k - 1][1, 0] * np.exp(
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


# Mat
class mat():

    def __init__(self, _lambda):
        self._lambda = _lambda
        self.i = complex(0, 1)

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
                                                                           (2 * np.sqrt(2) * a * sigma) * (
                                                                                   special.wofz(x) + special.wofz(
                                                                               y)))
        return eps

    def bk7(self):
        a = np.array(
            [190.75, 193.73, 196.8, 199.98, 203.25, 206.64, 210.14, 213.77, 217.52, 221.4, 225.43, 229.6, 233.93,
             238.43,
             243.11, 247.97, 253.03, 258.3, 263.8, 269.53, 275.52, 281.78, 288.34, 295.2, 302.4, 309.96, 317.91, 326.28,
             335.1, 344.4, 354.24, 364.66, 375.71, 387.45, 399.95, 413.28, 427.54, 442.8, 459.2, 476.87, 495.94, 516.6,
             539.0700000000001, 563.5700000000001, 590.41, 619.9299999999999, 652.55, 688.8099999999999,
             729.3200000000001, 774.91, 826.5700000000001, 885.61, 953.73, 1033.21, 1127.14, 1239.85])

        e = np.array([2.8406742849, 2.804261715649, 2.770563579001, 2.739329528464, 2.710376127684, 2.683571461921,
                      2.658732435844, 2.635706792196, 2.614339739664, 2.594480126121, 2.575996110081, 2.558755351321,
                      2.542647106624, 2.527572147556, 2.513432915161, 2.500146004225, 2.487638700625, 2.475842457361,
                      2.464692764356, 2.4541415649, 2.444132010384, 2.434623467584, 2.425575745476, 2.416958387649,
                      2.4087350401, 2.400878973529, 2.393369890704, 2.386181504529, 2.379296995009, 2.372696606736,
                      2.366369966601, 2.360300650929, 2.354472356041, 2.348881151236, 2.343513969316, 2.3383608889,
                      2.333415112704, 2.328666844009, 2.324109397009, 2.319733086489, 2.315534369344, 2.311503651769,
                      2.307628351744, 2.303905051044, 2.300324289124, 2.296867553764, 2.2935285136, 2.290285730161,
                      2.287120856329, 2.284009554436, 2.2809154729, 2.277793303696, 2.274585797929, 2.271202716601,
                      2.267514953929, 2.263330686969])

        eps = np.interp(self._lambda, a, e)
        return eps

    def cr(self):
        a = np.array(
            [206.64, 208.38, 210.14, 211.94, 213.77, 215.63, 217.52, 219.44, 221.4, 223.4, 225.43, 227.5, 229.6, 231.75,
             233.93, 236.16, 238.43, 240.75, 243.11, 245.52, 247.97, 250.48, 253.03, 255.64, 258.3, 261.02, 263.8,
             266.63,
             269.53, 272.49, 275.52, 278.62, 281.78, 285.02, 288.34, 291.73, 295.2, 298.76, 302.4, 306.14, 309.96,
             313.89,
             317.91, 322.04, 326.28, 330.63, 335.1, 339.69, 344.4, 349.25, 354.24, 359.38, 364.66, 370.11, 375.71,
             381.49,
             387.45, 393.6, 399.95, 406.51, 413.28, 420.29, 427.54, 435.04, 442.8, 450.86, 459.2, 467.87, 476.87,
             486.22,
             495.94, 506.06, 516.6, 527.6, 539.0700000000001, 551.05, 563.5700000000001, 576.6799999999999, 590.41,
             604.8099999999999, 619.9299999999999, 635.8200000000001, 652.55, 670.1900000000001, 688.8099999999999,
             708.49, 729.3200000000001, 751.4299999999999, 774.91, 799.9, 826.5700000000001, 855.0700000000001,
             885.61, 918.41, 953.73, 991.88, 1033.21, 1078.13, 1127.14, 1180.81, 1239.85])

        e = np.array([-0.7925 + 4.9932 * self.i, -0.8233535156249996 + 5.004690625 * self.i, -0.8268 + 5.0224 * self.i,
                      -0.8143253906250001 + 5.04143125 * self.i, -0.7974999999999999 + 5.0568 * self.i,
                      -0.7898343749999999 + 5.06141015625 * self.i, -0.7974999999999999 + 5.0568 * self.i,
                      -0.8382175781250001 + 5.039485937499999 * self.i,
                      -0.8904000000000001 + 5.016999999999999 * self.i,
                      -0.9344566406250001 + 4.995 * self.i, -0.9827000000000004 + 4.9764 * self.i,
                      -1.043912890625 + 4.968224999999999 * self.i, -1.1095 + 4.9632 * self.i,
                      -1.170687890625 + 4.954249999999999 * self.i, -1.2363 + 4.948399999999999 * self.i,
                      -1.314863671875 + 4.9485078125 * self.i, -1.3992 + 4.9594 * self.i,
                      -1.4845125 + 4.987915625 * self.i,
                      -1.5729 + 5.032000000000001 * self.i, -1.667080078125 + 5.093614062499999 * self.i,
                      -1.7604 + 5.168 * self.i, -1.844105859375001 + 5.252843749999999 * self.i,
                      -1.9256 + 5.343 * self.i,
                      -2.008280859375 + 5.433718750000001 * self.i, -2.0956 + 5.52 * self.i,
                      -2.198756640625 + 5.590724999999999 * self.i, -2.2981 + 5.657999999999999 * self.i,
                      -2.37359375 + 5.723549999999999 * self.i, -2.435999999999999 + 5.810199999999999 * self.i,
                      -2.482481640625 + 5.962537499999999 * self.i, -2.5347 + 6.1204 * self.i,
                      -2.6244109375 + 6.2189625 * self.i,
                      -2.7225 + 6.3072 * self.i, -2.806390624999999 + 6.413383593750001 * self.i,
                      -2.886000000000001 + 6.540800000000001 * self.i, -2.9661984375 + 6.713525 * self.i,
                      -3.039999999999999 + 6.899999999999999 * self.i, -3.098131640625 + 7.069900000000001 * self.i,
                      -3.150900000000001 + 7.238 * self.i, -3.208974609375 + 7.40703125 * self.i,
                      -3.263599999999999 + 7.584 * self.i, -3.304599999999999 + 7.784249999999999 * self.i,
                      -3.345600000000001 + 7.987 * self.i, -3.401062109375 + 8.17059375 * self.i,
                      -3.4611 + 8.35 * self.i,
                      -3.526905078125 + 8.530298437499999 * self.i, -3.5784 + 8.721 * self.i,
                      -3.583821484374999 + 8.9339625 * self.i,
                      -3.575199999999999 + 9.1686 * self.i, -3.571168750000001 + 9.453225000000002 * self.i,
                      -3.584000000000001 + 9.715200000000001 * self.i, -3.649484375 + 9.854289843749999 * self.i,
                      -3.722800000000001 + 9.969600000000002 * self.i, -3.759271874999999 + 10.13952734375 * self.i,
                      -3.788400000000002 + 10.336 * self.i, -3.820492187500001 + 10.55176875 * self.i,
                      -3.870400000000001 + 10.803 * self.i, -3.970114453125 + 11.1131546875 * self.i,
                      -4.082400000000002 + 11.457 * self.i, -4.174128124999998 + 11.80670859375 * self.i,
                      -4.258500000000001 + 12.1888 * self.i, -4.342965234375002 + 12.63259375 * self.i,
                      -4.4115 + 13.1068 * self.i,
                      -4.450119140625 + 13.573871875 * self.i, -4.457100000000001 + 14.074 * self.i,
                      -4.424873437500001 + 14.64025 * self.i, -4.352400000000001 + 15.264 * self.i,
                      -4.252692187500002 + 15.95943125 * self.i, -4.073999999999999 + 16.6912 * self.i,
                      -3.729884765625002 + 17.43339375 * self.i, -3.327499999999999 + 18.15 * self.i,
                      -3.000443359374998 + 18.772903125 * self.i, -2.620799999999999 + 19.3806 * self.i,
                      -2.026633984375 + 20.082478125 * self.i, -1.416800000000002 + 20.7126 * self.i,
                      -0.9634605468749982 + 21.1424765625 * self.i, -0.6539999999999981 + 21.3808 * self.i,
                      -0.5513378906250015 + 21.34546875 * self.i, -0.5858999999999988 + 21.186 * self.i,
                      -0.7052437500000011 + 21.02169921875 * self.i, -0.9043999999999972 + 20.856 * self.i,
                      -1.179668359375 + 20.732109375 * self.i, -1.478899999999999 + 20.646 * self.i,
                      -1.7591859375 + 20.610875 * self.i,
                      -1.993300000000001 + 20.6244 * self.i, -2.129018359374999 + 20.67811875 * self.i,
                      -2.196399999999999 + 20.808 * self.i, -2.213423828125 + 21.0548015625 * self.i,
                      -2.1615 + 21.3968 * self.i,
                      -1.997846484375 + 21.854225 * self.i, -1.8063 + 22.3416 * self.i,
                      -1.679373437499999 + 22.75875 * self.i,
                      -1.5663 + 23.1616 * self.i, -1.472354296874999 + 23.5625921875 * self.i,
                      -1.316699999999999 + 23.9944 * self.i,
                      -0.9806999999999988 + 24.5252125 * self.i, -0.5663999999999998 + 25.06 * self.i,
                      -0.1026824218749987 + 25.5120140625 * self.i * self.i, 0.2880000000000003 + 25.9192 * self.i,
                      0.5030988281249993 + 26.2924390625 * self.i, 0.4380000000000006 + 26.6432 * self.i])

        eps = np.interp(self._lambda, a, e)
        return eps

    def h2o(self):
        a = np.array(
            [404.7, 435.8, 467.8, 480, 508.5, 546.1, 577, 579.1, 589.1, 643.8, 700, 750, 800, 850, 900, 950, 1000, 1050,
             1100])
        e = np.array([1.8056640625, 1.7988442641, 1.7932691569, 1.7914216336, 1.7875957401, 1.7833999936, 1.7804632356,
                      1.7802764329, 1.7794226025, 1.7753164081, 1.7718005881, 1.766241, 1.763584, 1.760929, 1.763584,
                      1.760929,
                      1.758276, 1.755625, 1.752976])
        eps = np.interp(self._lambda, a, e)
        return eps

    def affichageEpsMat(self):
        print("Epsilon Au :", self.Aubb())
        print("Epsilon H2o :", self.h2o())
        print("Epsilon Cr :", self.cr())
        print("Epsilon BK7 : ", self.bk7())


# Haut de la page

st.set_page_config(page_title="Moosh", page_icon="./Images/appImage.png")


# Sidebar
class sidebarWidget:

    def sliderPara(self):
        n = st.slider("Nombre de miroirs", 1, 50, 20, 1)
        return n

    def checkPara(self):
        n = st.radio("Polarisation", ("Polarisation TE", "Polarisation TM"))
        return n

    def angleInput1(self):
        n = st.number_input("Angle d'incidence", 1, 90, 35, format=None, key=1)
        return n

    def angleInput2(self):
        n = st.number_input("Angle d'incidence", 0, 90, 35, format=None, key=2)
        return n

    def angleInput3(self):
        n = st.number_input("Angle d'incidence", 0, 90, 35, format=None, key=3)
        return n

    def lambdaInput1(self):
        n = st.number_input("Longueur d'onde", 400, 800, 600, format=None, key=1)
        return n

    def lambdaInput2(self):
        n = st.number_input("Longueur d'onde", 400, 800, 600, format=None, key=2)
        return n

    def lambdaInput3(self):
        n = st.number_input("Longueur d'onde", 400, 800, 600, format=None, key=3)
        return n

    def lambdaInput4(self):
        n = st.number_input("Longueur d'onde", 400, 800, 600, format=None, key=4)
        return n

    def beamPos(self):
        n = st.slider("Position", 0.0, 1.0, 0.4, 0.1)
        return n

    def selectBox1(self):
        n = st.selectbox("Choix du verre", ("BK7","BAF10","BAK1"))
        return n


# def homepage():
#     st.title("Moosh HomePage")
#     st.write('Homepage')
#     st.text('Work in progress')
#     st.text("Only the Bragg experiment is modeled => Moosh =>  Miroir de Bragg")
def homepage():
    st.write("Homepage")


def moosh():
    sideBarExp = st.sidebar.radio("Choix de l'expérience", ('Miroir de Bragg', 'Plasmon de surface', 'Photovoltaïque'))
    if sideBarExp == 'Miroir de Bragg':
        with st.sidebar.beta_expander(" Miroir de Bragg"):
            ####
            st.markdown(" ## Paramètres")
            mirpara = widget.sliderPara()
            polparab = widget.checkPara()
            ####
            st.markdown(" ## Coefficients")
            coefAng = widget.lambdaInput1()
            coefLamb = widget.angleInput1()
            btnCoef = st.button("Afficher les coefficients")
            ####
            st.markdown(" ## Angular")
            angLamb = widget.lambdaInput2()
            btnAng = st.button("Afficher Angular")
            ####
            st.markdown(" ## Spectrum")
            specAngle = widget.angleInput2()
            btnSpec = st.button("Afficher Spectrum")
            ####
            st.markdown(" ## Beam")
            bPos = widget.beamPos()
            beamLamb = widget.lambdaInput3()
            beamAng = widget.angleInput3()
            btnBeam = st.button("Afficher Beam")

        if polparab == 'Polarisation TE':
            polbragg = 1
        elif polpara == 'Polarisation TM':
            polbragg = 0

        Bragg_ = Bragg(mirpara, 200, polbragg)

        if btnCoef == 1:
            Bragg_.affichageCoef(coefAng, coefLamb)

        if btnAng == 1:
            Bragg_.angular(angLamb)

        if btnSpec == 1:
            Bragg_.spectrum(specAngle)

    elif sideBarExp == 'Plasmon de surface':
        st.text("SPR")
        st.text("Work in progress")
        with st.sidebar.beta_expander("Plasmon de surface"):
            ####
            lambmat = widget.lambdaInput4()
            material = mat(lambmat)
            ####
            st.markdown(" ## Choix des matériaux")
            ####
            sprVerre = widget.selectBox1()
            if sprVerre == 'BK7':
                epsVerre = material.bk7()
            st.write("Espilon du verre =", epsVerre)


def documentation():
    st.write('Docs')
    st.text("Work in progress")


widget = sidebarWidget()

st.sidebar.title("Navigation")
st.sidebar.write('')

side_menu_navigation = st.sidebar.radio('', ('Homepage', 'Moosh', 'Documentation'))
if side_menu_navigation == 'Homepage':
    homepage()
elif side_menu_navigation == 'Moosh':
    moosh()
elif side_menu_navigation == 'Documentation':
    documentation()

