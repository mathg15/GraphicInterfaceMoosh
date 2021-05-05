import numpy as np
import altair as alt
import matplotlib.pyplot as plt
import streamlit as st
import matplotlib.image as im

# Bragg

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

# Haut de la page

st.set_page_config(page_title="Moosh", page_icon="./Images/appImage.png")

# Sidebar
class sidebarWidget:

    def sliderPara(self):
        n = st.slider("Nombre de miroirs", 1, 50, 20, 1)
        return n

    def angleInput1(self):
        n = st.number_input("Angle d'incidence", 0, 90, 35,format=None,key=1)
        return n

    def angleInput2(self):
        n = st.number_input("Angle d'incidence", 0, 90, 35, format=None, key=2)
        return n

    def angleInput3(self):
        n = st.number_input("Angle d'incidence", 0, 90, 35, format=None, key=3)
        return n

    def lambdaInput1(self):
        n = st.number_input("Longueur d'onde", 400, 800, 600,format=None,key=1)
        return n

    def lambdaInput2(self):
        n = st.number_input("Longueur d'onde", 400, 800, 600,format=None,key=2)
        return n

    def lambdaInput3(self):
        n = st.number_input("Longueur d'onde", 400, 800, 600,format=None,key=3)
        return n

    def beamPos(self):
        n = st.slider("Position",0.0,1.0,0.4,0.1)
        return n

widget = sidebarWidget()


with st.sidebar.beta_expander(" Miroir de Bragg"):
    ####
    st.markdown(" ## Paramètres")
    mirpara = widget.sliderPara()
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

Bragg_ = Bragg(mirpara,200)

if btnCoef == 1:
    Bragg_.affichageCoef(coefAng, coefLamb)

if btnAng == 1:
    Bragg_.angular(angLamb)

if btnSpec == 1:
    Bragg_.spectrum(specAngle)
