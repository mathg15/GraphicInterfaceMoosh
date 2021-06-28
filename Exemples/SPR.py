import numpy as np
import matplotlib.pyplot as plt
from scipy import special
import streamlit as st
import matplotlib.image as im
import matplotlib.colors as mcolors

from Tools.mat import *


class SPR:

    def __init__(self, pol):

        self.nbDeCouches = 4

        if pol == "TM":
            polw = 0
        elif pol == "TE":
            polw = 1

        self.pol = polw

    def typemat(self, _lambda):

        m = mat(_lambda)

        Eps1 = m.bk7()
        Eps2 = m.cr()
        Eps3 = m.Au()
        Eps4 = 1

        return Eps1, Eps2, Eps3, Eps4

    def structure(self, _lambda):

        Eps1, Eps2, Eps3, Eps4 = self.typemat(_lambda)

        Eps = np.array([Eps1, Eps2, Eps3, Eps4])
        Mu = np.array([1, 1, 1, 1])
        Type = np.array([0, 1, 2, 3])
        hauteur = np.array([4000, 4, 38, 1000])

        return Eps, Mu, Type, hauteur

    def cascade(self, A, B):
        # On combine deux matrices de diffusion A et B en une seule matrice de diffusion (2*2) S
        t = 1 / (1 - B[0, 0] * A[1, 1])
        S = np.array([[A[0, 0] + A[0, 1] * B[0, 0] * A[1, 0] * t, A[0, 1] * B[0, 1] * t],
                      [B[1, 0] * A[1, 0] * t, B[1, 1] + A[1, 1] * B[0, 1] * B[1, 0] * t]], dtype=complex)

        return S

    def coefficient(self, thetacoef, _lambda):

        TypeEps, TypeMu, Type, hauteur = self.structure(_lambda)

        # On considère que la première valeur de la hauteur est 0
        hauteur[0] = 0

        # On definie k0 à partir de lambda
        k0 = (2 * np.pi) / _lambda

        # g est la longeur de Type
        g = len(Type)

        # En fonction de la polarisation, f prend soit la valeur TypeMu ou la valeur TypeEps
        if self.pol == 0:
            f = TypeMu
        elif self.pol == 1:
            f = TypeEps

        # Définition de alpha et gamma en fonction de TypeEps, TypeMu, k0 et Theta
        alpha = np.sqrt(TypeEps[0] * TypeMu[0]) * k0 * np.sin(thetacoef * (np.pi / 180))
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
        T = np.zeros((2 * g, 2, 2), dtype=complex)
        T[0] = [[0, 1], [1, 0]]

        # Cacul des matrices S
        for k in range(g - 1):
            # Matrice de diffusion des couches
            t = np.exp(1j * gamma[k] * hauteur[k])
            T[2 * k + 1] = np.array([[0, t], [t, 0]])

            # Matrice de diffusion d'interface
            b1 = gamma[k] / (f[k])
            b2 = gamma[k + 1] / (f[k + 1])
            T[2 * k + 2] = [[(b1 - b2) / (b1 + b2), (2 * b2) / (b1 + b2)],
                            [(2 * b1) / (b1 + b2), (b2 - b1) / (b1 + b2)]]

        # Matrice de diffusion pour la dernière couche
        t = np.exp(1j * gamma[g - 1] * hauteur[g - 1])
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

        Npoints = 200
        # Intervalle angulaire
        maxAngle = 30
        minAngle = 60
        rangeAngle = np.linspace(minAngle, maxAngle, Npoints)

        # Création des matrices
        a = np.ones((Npoints, 1), dtype=complex)
        b = np.ones((Npoints, 1), dtype=complex)
        c = np.ones((Npoints, 1), dtype=complex)
        d = np.ones((Npoints, 1), dtype=complex)

        for i in range(Npoints):
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

        Npoints = 200
        # Intervalle de longeur d'onde
        minLambda = 400
        maxLambda = 700
        rangeLambda = np.linspace(minLambda, maxLambda, Npoints)

        a = np.ones((Npoints, 1), dtype=complex)
        b = np.ones((Npoints, 1), dtype=complex)
        c = np.ones((Npoints, 1), dtype=complex)
        d = np.ones((Npoints, 1), dtype=complex)

        for i in range(Npoints):
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

        TypeEps, TypeMu, Type, hauteur = self.structure(_lambda)

        _theta = _theta * (np.pi / 180)

        # Spatial window size
        d = 70 * _lambda

        # Incident beam width
        w = 10 * _lambda

        # Number of pixels horizontally
        nx = np.floor(d / 10)

        # Number of pixels verticaly
        ny = np.floor(hauteur / 10)
        # print(ny)
        # Number of modes
        nmod = np.floor(0.83660 * (d / w))

        hauteur = hauteur / d
        l = _lambda / d
        w = w / d

        if self.pol == 0:
            f = TypeEps
        elif self.pol == 1:
            f = TypeMu

        k0 = (2 * np.pi) / l

        En = np.zeros((int(np.sum(ny)), int(nx)), dtype=complex)

        # g est la longeur de Type
        g = len(Type)

        # Amplitude of the different modes
        X = np.exp(-(w ** 2) * (np.pi ** 2) * np.arange(-nmod, nmod + 1) ** 2) * np.exp(
            2 * 1j * np.pi * np.arange(-nmod, nmod + 1) * C)

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
                t = np.exp(1j * gamma[k] * hauteur[k])
                T[2 * k + 1] = np.array([[0, t], [t, 0]])

                # Matrice de diffusion d'interface
                b1 = gamma[k] / (f[k])
                b2 = gamma[k + 1] / (f[k + 1])
                T[2 * k + 2] = [[(b1 - b2) / (b1 + b2), (2 * b2) / (b1 + b2)],
                                [(2 * b1) / (b1 + b2), (b2 - b1) / (b1 + b2)]]

            # Matrice de diffusion pour la dernière couche
            t = np.exp(1j * gamma[g - 1] * hauteur[g - 1])
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
                I[j] = np.array([[A[j][1, 0], A[j][1, 1] * H[len(T) - j - 2][0, 1]],
                                 [A[j][1, 0] * H[len(T) - j - 2][0, 0], H[len(T) - j - 2][0, 1]]] / (
                                        1 - A[j][1, 1] * H[len(T) - j - 2][0, 0]))
            h = 0
            t = 0

            E = np.zeros((int(np.sum(ny)), 1), dtype=complex)

            for k in range(g):
                for m in range(int(ny[k])):
                    h = h + hauteur[k] / ny[k]
                    E[t, 0] = I[2 * k][0, 0] * np.exp(1j * gamma[k] * h) + I[2 * k + 1][1, 0] * np.exp(
                        1j * gamma[k] * (hauteur[k] - h))
                    t += 1
                h = 0

            E = E * np.exp(1j * alpha * np.arange(0, nx) / nx)
            En = En + X[int(nm)] * E

        V = np.abs(En)
        V = V / V.max()

        norm = mcolors.Normalize(vmax=V.max(), vmin=V.min())

        plt.figure(1)
        plt.pcolormesh(V / V.max(), norm=norm, cmap='jet')
        plt.colorbar()
        st.pyplot(plt)
