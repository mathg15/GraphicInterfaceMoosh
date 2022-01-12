import numpy as np
import matplotlib.pyplot as plt
from scipy import special
import streamlit as st
import matplotlib.image as im
import matplotlib.colors as mcolors
from Tools.mat import *
from Tools.widget import *

widget = sidebarWidget()


class mooshGen:

    def __init__(self, nbCouche, pol):
        self.nombreCouches = nbCouche

        self.Type = np.array([], dtype=complex)
        self.hauteur = np.array([])

        if pol == "Polarisation TE":
            self.pol = 1
        elif pol == "Polarisation TM":
            self.pol = 0

        

        if self.nombreCouches == 1:
            st.write("Couche 1")
            self.mat1 = widget.selecBoxGen1()
            self.haut1 = widget.hauteurGen1()

        elif self.nombreCouches == 2:
            st.write("Couche 1")
            self.mat1 = widget.selecBoxGen1()
            self.haut1 = widget.hauteurGen1()
            st.write("Couche 2")
            self.mat2 = widget.selecBoxGen2()
            self.haut2 = widget.hauteurGen2()

        elif self.nombreCouches == 3:
            st.write("Couche 1")
            self.mat1 = widget.selecBoxGen1()
            self.haut1 = widget.hauteurGen1()
            st.write("Couche 2")
            self.mat2 = widget.selecBoxGen2()
            self.haut2 = widget.hauteurGen2()
            st.write("Couche 3")
            self.mat3 = widget.selecBoxGen3()
            self.haut3 = widget.hauteurGen3()

        elif self.nombreCouches == 4:
            st.write("Couche 1")
            self.mat1 = widget.selecBoxGen1()
            self.haut1 = widget.hauteurGen1()
            st.write("Couche 2")
            self.mat2 = widget.selecBoxGen2()
            self.haut2 = widget.hauteurGen2()
            st.write("Couche 3")
            self.mat3 = widget.selecBoxGen3()
            self.haut3 = widget.hauteurGen3()
            st.write("Couche 4")
            self.mat4 = widget.selecBoxGen4()
            self.haut4 = widget.hauteurGen4()

        elif self.nombreCouches == 5:
            st.write("Couche 1")
            self.mat1 = widget.selecBoxGen1()
            self.haut1 = widget.hauteurGen1()
            st.write("Couche 2")
            self.mat2 = widget.selecBoxGen2()
            self.haut2 = widget.hauteurGen2()
            st.write("Couche 3")
            self.mat3 = widget.selecBoxGen3()
            self.haut3 = widget.hauteurGen3()
            st.write("Couche 4")
            self.mat4 = widget.selecBoxGen4()
            self.haut4 = widget.hauteurGen4()
            st.write("Couche 5")
            self.mat5 = widget.selecBoxGen5()
            self.haut5 = widget.hauteurGen5()

        elif self.nombreCouches == 6:
            st.write("Couche 1")
            self.mat1 = widget.selecBoxGen1()
            self.haut1 = widget.hauteurGen1()
            st.write("Couche 2")
            self.mat2 = widget.selecBoxGen2()
            self.haut2 = widget.hauteurGen2()
            st.write("Couche 3")
            self.mat3 = widget.selecBoxGen3()
            self.haut3 = widget.hauteurGen3()
            st.write("Couche 4")
            self.mat4 = widget.selecBoxGen4()
            self.haut4 = widget.hauteurGen4()
            st.write("Couche 5")
            self.mat5 = widget.selecBoxGen5()
            self.haut5 = widget.hauteurGen5()
            st.write("Couche 6")
            self.mat6 = widget.selecBoxGen6()
            self.haut6 = widget.hauteurGen6()

        elif self.nombreCouches == 7:
            st.write("Couche 1")
            self.mat1 = widget.selecBoxGen1()
            self.haut1 = widget.hauteurGen1()
            st.write("Couche 2")
            self.mat2 = widget.selecBoxGen2()
            self.haut2 = widget.hauteurGen2()
            st.write("Couche 3")
            self.mat3 = widget.selecBoxGen3()
            self.haut3 = widget.hauteurGen3()
            st.write("Couche 4")
            self.mat4 = widget.selecBoxGen4()
            self.haut4 = widget.hauteurGen4()
            st.write("Couche 5")
            self.mat5 = widget.selecBoxGen5()
            self.haut5 = widget.hauteurGen5()
            st.write("Couche 6")
            self.mat6 = widget.selecBoxGen6()
            self.haut6 = widget.hauteurGen6()
            st.write("Couche 7")
            self.mat7 = widget.selecBoxGen7()
            self.haut7 = widget.hauteurGen7()

        elif self.nombreCouches == 8:
            st.write("Couche 1")
            self.mat1 = widget.selecBoxGen1()
            self.haut1 = widget.hauteurGen1()
            st.write("Couche 2")
            self.mat2 = widget.selecBoxGen2()
            self.haut2 = widget.hauteurGen2()
            st.write("Couche 3")
            self.mat3 = widget.selecBoxGen3()
            self.haut3 = widget.hauteurGen3()
            st.write("Couche 4")
            self.mat4 = widget.selecBoxGen4()
            self.haut4 = widget.hauteurGen4()
            st.write("Couche 5")
            self.mat5 = widget.selecBoxGen5()
            self.haut5 = widget.hauteurGen5()
            st.write("Couche 6")
            self.mat6 = widget.selecBoxGen6()
            self.haut6 = widget.hauteurGen6()
            st.write("Couche 7")
            self.mat7 = widget.selecBoxGen7()
            self.haut7 = widget.hauteurGen7()
            st.write("Couche 8")
            self.mat8 = widget.selecBoxGen8()
            self.haut8 = widget.hauteurGen8()

        elif self.nombreCouches == 9:
            st.write("Couche 1")
            self.mat1 = widget.selecBoxGen1()
            self.haut1 = widget.hauteurGen1()
            st.write("Couche 2")
            self.mat2 = widget.selecBoxGen2()
            self.haut2 = widget.hauteurGen2()
            st.write("Couche 3")
            self.mat3 = widget.selecBoxGen3()
            self.haut3 = widget.hauteurGen3()
            st.write("Couche 4")
            self.mat4 = widget.selecBoxGen4()
            self.haut4 = widget.hauteurGen4()
            st.write("Couche 5")
            self.mat5 = widget.selecBoxGen5()
            self.haut5 = widget.hauteurGen5()
            st.write("Couche 6")
            self.mat6 = widget.selecBoxGen6()
            self.haut6 = widget.hauteurGen6()
            st.write("Couche 7")
            self.mat7 = widget.selecBoxGen7()
            self.haut7 = widget.hauteurGen7()
            st.write("Couche 8")
            self.mat8 = widget.selecBoxGen8()
            self.haut8 = widget.hauteurGen8()
            st.write("Couche 9")
            self.mat9 = widget.selecBoxGen9()
            self.haut9 = widget.hauteurGen9()

        elif self.nombreCouches == 10:
            st.write("Couche 1")
            self.mat1 = widget.selecBoxGen1()
            self.haut1 = widget.hauteurGen1()
            st.write("Couche 2")
            self.mat2 = widget.selecBoxGen2()
            self.haut2 = widget.hauteurGen2()
            st.write("Couche 3")
            self.mat3 = widget.selecBoxGen3()
            self.haut3 = widget.hauteurGen3()
            st.write("Couche 4")
            self.mat4 = widget.selecBoxGen4()
            self.haut4 = widget.hauteurGen4()
            st.write("Couche 5")
            self.mat5 = widget.selecBoxGen5()
            self.haut5 = widget.hauteurGen5()
            st.write("Couche 6")
            self.mat6 = widget.selecBoxGen6()
            self.haut6 = widget.hauteurGen6()
            st.write("Couche 7")
            self.mat7 = widget.selecBoxGen7()
            self.haut7 = widget.hauteurGen7()
            st.write("Couche 8")
            self.mat8 = widget.selecBoxGen8()
            self.haut8 = widget.hauteurGen8()
            st.write("Couche 9")
            self.mat9 = widget.selecBoxGen9()
            self.haut9 = widget.hauteurGen9()
            st.write("Couche 10")
            self.mat10 = widget.selecBoxGen10()
            self.haut10 = widget.hauteurGen10()

    def typemat(self, lam):

        m = mat(lam)
        epsbk7 = m.bk7()
        epsCr = m.cr()
        epsAu = m.Au()
        epsH2O = m.h2o()
        epsTiO2 = m.TiO2()
        epsSiO2 = m.SiO2()
        epsAg = m.Ag()
        epsZnO = m.ZnO()
        ####

        if self.mat1 == 'Air':
            Eps1 = 1
        elif self.mat1 == 'Eau':
            Eps1 = epsH2O
        elif self.mat1 == 'Bk7':
            Eps1 = epsbk7
        elif self.mat1 == 'SiO2':
            Eps1 = epsSiO2
        elif self.mat1 == 'TiO2':
            Eps1 = epsTiO2
        elif self.mat1 == 'Au':
            Eps1 = epsAu
        elif self.mat1 == 'Cr':
            Eps1 = epsCr
        elif self.mat1 == 'Ag':
            Eps1 = epsAg
        elif self.mat1 == 'ZnO':
            Eps1 = epsZnO
        if self.nombreCouches == 1:
            return Eps1

        ####

        if self.mat2 == 'Air':
            Eps2 = 1
        elif self.mat2 == 'Eau':
            Eps2 = epsH2O
        elif self.mat2 == 'Bk7':
            Eps2 = epsbk7
        elif self.mat2 == 'SiO2':
            Eps2 = epsSiO2
        elif self.mat2 == 'TiO2':
            Eps2 = epsTiO2
        elif self.mat2 == 'Au':
            Eps2 = epsAu
        elif self.mat2 == 'Cr':
            Eps2 = epsCr
        elif self.mat2 == 'Ag':
            Eps2 = epsAg
        elif self.mat2 == 'ZnO':
            Eps2 = epsZnO

        if self.nombreCouches == 2:
            return Eps1, Eps2

        ####

        if self.mat3 == 'Air':
            Eps3 = 1
        elif self.mat3 == 'Eau':
            Eps3 = epsH2O
        elif self.mat3 == 'Bk7':
            Eps3 = epsbk7
        elif self.mat3 == 'SiO2':
            Eps3 = epsSiO2
        elif self.mat3 == 'TiO2':
            Eps3 = epsTiO2
        elif self.mat3 == 'Au':
            Eps3 = epsAu
        elif self.mat3 == 'Cr':
            Eps3 = epsCr
        elif self.mat3 == 'Ag':
            Eps3 = epsAg
        elif self.mat3 == 'ZnO':
            Eps3 = epsZnO

        if self.nombreCouches == 3:
            return Eps1, Eps2, Eps3

        ####

        if self.mat4 == 'Air':
            Eps4 = 1
        elif self.mat4 == 'Eau':
            Eps4 = epsH2O
        elif self.mat4 == 'Bk7':
            Eps4 = epsbk7
        elif self.mat4 == 'SiO2':
            Eps4 = epsSiO2
        elif self.mat4 == 'TiO2':
            Eps4 = epsTiO2
        elif self.mat4 == 'Au':
            Eps4 = epsAu
        elif self.mat4 == 'Cr':
            Eps4 = epsCr
        elif self.mat4 == 'Ag':
            Eps4 = epsAg
        elif self.mat4 == 'ZnO':
            Eps4 = epsZnO

        if self.nombreCouches == 4:
            return Eps1, Eps2, Eps3, Eps4

        ####

        if self.mat5 == 'Air':
            Eps5 = 1
        elif self.mat5 == 'Eau':
            Eps5 = epsH2O
        elif self.mat5 == 'Bk7':
            Eps5 = epsbk7
        elif self.mat5 == 'SiO2':
            Eps5 = epsSiO2
        elif self.mat5 == 'TiO2':
            Eps5 = epsTiO2
        elif self.mat5 == 'Au':
            Eps5 = epsAu
        elif self.mat5 == 'Cr':
            Eps5 = epsCr
        elif self.mat5 == 'Ag':
            Eps5 = epsAg
        elif self.mat5 == 'ZnO':
            Eps5 = epsZnO

        if self.nombreCouches == 5:
            return Eps1, Eps2, Eps3, Eps4, Eps5

        ####

        if self.mat6 == 'Air':
            Eps6 = 1
        elif self.mat6 == 'Eau':
            Eps6 = epsH2O
        elif self.mat6 == 'Bk7':
            Eps6 = epsbk7
        elif self.mat6 == 'SiO2':
            Eps6 = epsSiO2
        elif self.mat6 == 'TiO2':
            Eps6 = epsTiO2
        elif self.mat6 == 'Au':
            Eps6 = epsAu
        elif self.mat6 == 'Cr':
            Eps6 = epsCr
        elif self.mat6 == 'Ag':
            Eps6 = epsAg
        elif self.mat6 == 'ZnO':
            Eps6 = epsZnO

        if self.nombreCouches == 6:
            return Eps1, Eps2, Eps3, Eps4, Eps5, Eps6

        ####

        if self.mat7 == 'Air':
            Eps7 = 1
        elif self.mat7 == 'Eau':
            Eps7 = epsH2O
        elif self.mat7 == 'Bk7':
            Eps7 = epsbk7
        elif self.mat7 == 'SiO2':
            Eps7 = epsSiO2
        elif self.mat7 == 'TiO2':
            Eps7 = epsTiO2
        elif self.mat7 == 'Au':
            Eps7 = epsAu
        elif self.mat7 == 'Cr':
            Eps7 = epsCr
        elif self.mat7 == 'Ag':
            Eps7 = epsAg
        elif self.mat7 == 'ZnO':
            Eps7 = epsZnO

        if self.nombreCouches == 7:
            return Eps1, Eps2, Eps3, Eps4, Eps5, Eps6, Eps7

        ####

        if self.mat8 == 'Air':
            Eps8 = 1
        elif self.mat8 == 'Eau':
            Eps8 = epsH2O
        elif self.mat8 == 'Bk7':
            Eps8 = epsbk7
        elif self.mat8 == 'SiO2':
            Eps8 = epsSiO2
        elif self.mat8 == 'TiO2':
            Eps8 = epsTiO2
        elif self.mat8 == 'Au':
            Eps8 = epsAu
        elif self.mat8 == 'Cr':
            Eps8 = epsCr
        elif self.mat8 == 'Ag':
            Eps8 = epsAg
        elif self.mat8 == 'ZnO':
            Eps8 = epsZnO

        if self.nombreCouches == 8:
            return Eps1, Eps2, Eps3, Eps4, Eps5, Eps6, Eps7, Eps8

        ####

        if self.mat9 == 'Air':
            Eps9 = 1
        elif self.mat9 == 'Eau':
            Eps9 = epsH2O
        elif self.mat9 == 'Bk7':
            Eps9 = epsbk7
        elif self.mat9 == 'SiO2':
            Eps9 = epsSiO2
        elif self.mat9 == 'TiO2':
            Eps9 = epsTiO2
        elif self.mat9 == 'Au':
            Eps9 = epsAu
        elif self.mat9 == 'Cr':
            Eps9 = epsCr
        elif self.mat9 == 'Ag':
            Eps9 = epsAg
        elif self.mat9 == 'ZnO':
            Eps9 = epsZnO

        if self.nombreCouches == 9:
            return Eps1, Eps2, Eps3, Eps4, Eps5, Eps6, Eps7, Eps8, Eps9

        ####

        if self.mat10 == 'Air':
            Eps10 = 1
        elif self.mat10 == 'Eau':
            Eps10 = epsH2O
        elif self.mat10 == 'Bk7':
            Eps10 = epsbk7
        elif self.mat10 == 'SiO2':
            Eps10 = epsSiO2
        elif self.mat10 == 'TiO2':
            Eps10 = epsTiO2
        elif self.mat10 == 'Au':
            Eps10 = epsAu
        elif self.mat10 == 'Cr':
            Eps10 = epsCr
        elif self.mat10 == 'Ag':
            Eps10 = epsAg
        elif self.mat10 == 'ZnO':
            Eps10 = epsZnO

        if self.nombreCouches == 10:
            return Eps1, Eps2, Eps3, Eps4, Eps5, Eps6, Eps7, Eps8, Eps9, Eps10

    def structure(self, lam):
        if self.nombreCouches == 1:
            Eps1 = self.typemat(lam)
            Eps = np.array([Eps1], dtype=complex)
            Mu = np.array([1])
            Type = np.array([0], dtype=complex)
            hauteur = np.array([self.haut1])
            return Eps, Mu, Type, hauteur

        elif self.nombreCouches == 2:
            Eps1, Eps2 = self.typemat(lam)
            Eps = np.array([Eps1, Eps2], dtype=complex)
            Mu = np.array([1, 1])
            hauteur = np.array([self.haut1, self.haut2])
            Type = np.array([0, 1], dtype=complex)
            return Eps, Mu, Type, hauteur

        elif self.nombreCouches == 3:
            Eps1, Eps2, Eps3 = self.typemat(lam)
            Eps = np.array([Eps1, Eps2, Eps3], dtype=complex)
            Mu = np.array([1, 1, 1])
            hauteur = np.array([self.haut1, self.haut2, self.haut3])
            Type = np.array([0, 1, 2], dtype=complex)
            return Eps, Mu, Type, hauteur

        elif self.nombreCouches == 4:
            Eps1, Eps2, Eps3, Eps4 = self.typemat(lam)
            Eps = np.array([Eps1, Eps2, Eps3, Eps4], dtype=complex)
            Mu = np.array([1, 1, 1, 1])
            hauteur = np.array([self.haut1, self.haut2, self.haut3, self.haut4])
            Type = np.array([0, 1, 2, 3], dtype=complex)
            return Eps, Mu, Type, hauteur

        elif self.nombreCouches == 5:
            Eps1, Eps2, Eps3, Eps4, Eps5 = self.typemat(lam)
            Eps = np.array([Eps1, Eps2, Eps3, Eps4, Eps5], dtype=complex)
            Mu = np.array([1, 1, 1, 1, 1])
            hauteur = np.array([self.haut1, self.haut2, self.haut3, self.haut4, self.haut5])
            Type = np.array([0, 1, 2, 3, 4], dtype=complex)
            return Eps, Mu, Type, hauteur

        elif self.nombreCouches == 6:
            Eps1, Eps2, Eps3, Eps4, Eps5, Eps6 = self.typemat(lam)
            Eps = np.array([Eps1, Eps2, Eps3, Eps4, Eps5, Eps6], dtype=complex)
            Mu = np.array([1, 1, 1, 1, 1, 1])
            hauteur = np.array([self.haut1, self.haut2, self.haut3, self.haut4, self.haut5, self.haut6])
            Type = np.array([0, 1, 2, 3, 4, 5], dtype=complex)
            return Eps, Mu, Type, hauteur

        elif self.nombreCouches == 7:
            Eps1, Eps2, Eps3, Eps4, Eps5, Eps6, Eps7 = self.typemat(lam)
            Eps = np.array([Eps1, Eps2, Eps3, Eps4, Eps5, Eps6, Eps7], dtype=complex)
            Mu = np.array([1, 1, 1, 1, 1, 1, 1])
            hauteur = np.array(
                [self.haut1, self.haut2, self.haut3, self.haut4, self.haut5, self.haut6, self.haut7])
            self.Type = np.array([0, 1, 2, 3, 4, 5, 6], dtype=complex)
            return Eps, Mu, Type, hauteur

        elif self.nombreCouches == 8:
            Eps1, Eps2, Eps3, Eps4, Eps5, Eps6, Eps7, Eps8 = self.typemat(lam)
            Eps = np.array([Eps1, Eps2, Eps3, Eps4, Eps5, Eps6, Eps7, Eps8], dtype=complex)
            Mu = np.array([1, 1, 1, 1, 1, 1, 1, 1])
            hauteur = np.array([self.haut1, self.haut2, self.haut3, self.haut4, self.haut5, self.haut6, self.haut7,
                                self.haut8])
            Type = np.array([0, 1, 2, 3, 4, 5, 6, 7], dtype=complex)
            return Eps, Mu, Type, hauteur

        elif self.nombreCouches == 9:
            Eps1, Eps2, Eps3, Eps4, Eps5, Eps6, Eps7, Eps8, Eps9 = self.typemat(lam)
            Eps = np.array([Eps1, Eps2, Eps3, Eps4, Eps5, Eps6, Eps7, Eps8, Eps9], dtype=complex)
            Mu = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1])
            hauteur = np.array([self.haut1, self.haut2, self.haut3, self.haut4, self.haut5, self.haut6, self.haut7,
                                self.haut8, self.haut9])
            Type = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8], dtype=complex)
            return Eps, Mu, Type, hauteur

        elif self.nombreCouches == 10:
            Eps1, Eps2, Eps3, Eps4, Eps5, Eps6, Eps7, Eps8, Eps9, Eps10 = self.typemat(lam)
            Eps = np.array([Eps1, Eps2, Eps3, Eps4, Eps5, Eps6, Eps7, Eps8, Eps9, Eps10], dtype=complex)
            Mu = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
            hauteur = np.array([self.haut1, self.haut2, self.haut3, self.haut4, self.haut5, self.haut6, self.haut7,
                                self.haut8, self.haut9, self.haut10])
            Type = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=complex)
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

        if self.pol == 1:
            f = TypeMu
        elif self.pol == 0:
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
            gamma[g - 1] = -np.sqrt(TypeEps[g - 1] * TypeMu[g - 1] * (k0 ** 2) - (alpha ** 2))
        else:
            gamma[g - 1] = np.sqrt(TypeEps[g - 1] * TypeMu[g - 1] * (k0 ** 2) - (alpha ** 2))

        # Définition de la matrice Scattering
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
        st.write("Coefficient de reflexion de l'énergie :", reflexionE)
        st.write("Coefficient de transmission de l'énergie :", transmissionE)

    def angular(self, lambda_, switch, minAngle, maxAngle):

        # Intervalle angulaire

        # maxAngle = 60
        # minAngle = 30
        rangeAngle = np.linspace(minAngle, maxAngle, 200)

        # Création des matrices
        a = np.ones((200, 1), dtype=complex)
        b = np.ones((200, 1), dtype=complex)
        c = np.ones((200, 1), dtype=complex)
        d = np.ones((200, 1), dtype=complex)

        for i in range(200):
            tht = rangeAngle[i]
            a[i], b[i], c[i], d[i] = self.coefficient(tht, lambda_)

        reflectMin = np.min(abs(c))

        if switch == "Reflexion":

            plt.figure(1)
            plt.subplot(211)
            plt.title(f"Reflection for lambda = {lambda_} \n Min relfec = {reflectMin}")
            plt.plot(rangeAngle, abs(c))
            plt.plot(1, reflectMin)
            plt.ylabel("Reflection")
            plt.xlabel("Angle (degrees)")
            plt.grid(True)
            plt.subplot(212)
            plt.plot(rangeAngle, np.angle(a))
            plt.ylabel("Phase")
            plt.xlabel("Angle")
            plt.title("Phase of the reflection coefficient")
            plt.tight_layout()
            st.pyplot(plt)

        elif switch == "Transmission":

            plt.figure(1)
            plt.subplot(211)
            plt.title("Transmission")
            plt.plot(rangeAngle, abs(d))
            plt.ylabel("Transmission")
            plt.xlabel("Angle (degrees)")
            plt.subplot(212)
            plt.plot(rangeAngle, np.angle(b))
            plt.ylabel("Phase")
            plt.xlabel("Angle")
            plt.title("Phase of the Transmission coefficient")
            plt.tight_layout()
            fig.canvas.mpl_connect('pick_event', on_pick)
            st.pyplot(plt)

    def spectrum(self, theta_):
        minLambda = 400
        maxLambda = 800
        rangeLambda = np.linspace(minLambda, maxLambda, 200)

        a = np.ones((200, 1), dtype=complex)
        b = np.ones((200, 1), dtype=complex)
        c = np.ones((200, 1), dtype=complex)
        d = np.ones((200, 1), dtype=complex)

        for i in range(200):
            rL = rangeLambda[i]
            a[i], b[i], c[i], d[i] = self.coefficient(theta_, rL)

        plt.figure(1)
        plt.subplot(211)
        plt.title("Reflexion for lambda ")
        plt.plot(rangeLambda, abs(c))
        plt.ylabel("Reflexion")
        plt.xlabel("Angle (degrees)")
        plt.subplot(212)
        plt.plot(rangeLambda, np.angle(a))
        plt.ylabel("Phase")
        plt.xlabel("Angle")
        plt.title("Phase of the reflection coefficient")
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

        if self.pol == 1:
            f = TypeEps
        elif self.pol == 0:
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
        V = np.flip(V)
        V = V / V.max()

        theta = _theta * (180 / np.pi)
        norm = mcolors.Normalize(vmax=V.max(), vmin=V.min())

        plt.figure(1)
        plt.pcolormesh(V / V.max(), norm=norm, cmap='jet')
        plt.colorbar()
        plt.title(f"Light beam for lambda = {_lambda} \n with an incidence angle of {theta} degrees")
        st.pyplot(plt)

    # def absorption(self, _theta, _lambda):
    #
    #     Eps, Mu, Type, hauteur = self.structure(_lambda)
    #
    #     hauteur[0] = 0
    #
    #     if self.pol == 0:
    #         f = Mu
    #     elif self.pol == 1:
    #         f = Eps
    #
    #     k0 = 2 * np.pi / _lambda
    #
    #     g = len(Type)
    #
    #     alpha = np.sqrt(Eps[0] * Mu[0]) * k0 * np.sin(thetacoef * (np.pi / 180))
    #     gamma = np.sqrt(Eps * Mu * (k0 ** 2) - np.ones(g) * (alpha ** 2))
    #
    #     # On fait en sorte d'avoir un résultat positif en foction au cas où l'index serait négatif
    #     if np.real(Eps[0]) < 0 and np.real(Mu[0]):
    #         gamma[0] = -gamma[0]
    #
    #     # On modifie la détermination de la racine carrée pour obtenir un stabilité parfaite
    #     if g > 2:
    #         gamma[1: g - 2] = gamma[1:g - 2] * (1 - 2 * (np.imag(gamma[1:g - 2]) < 0))
    #     # Condition de l'onde sortante pour le dernier milieu
    #     if np.real(Eps[g - 1]) < 0 and np.real(Mu[g - 1]) < 0 and np.real(
    #             np.sqrt(Eps[g - 1] * Mu * (k0 ** 2) - (alpha ** 2))):
    #         gamma[g - 1] = -np.sqrt(Eps[g - 1] * Mu[g - 1] * (k0 ** 2) - (alpha ** 2))
    #     else:
    #         gamma[g - 1] = np.sqrt(Eps[g - 1] * Mu[g - 1] * (k0 ** 2) - (alpha ** 2))
    #
    #     T = np.zeros((2 * g, 2, 2), dtype=complex)
    #     T[0] = [[0, 1], [1, 0]]
    #
    #     # Cacul des matrices S
    #     for k in range(g - 1):
    #         # Matrice de diffusion des couches
    #         t = np.exp(1j * gamma[k] * hauteur[k])
    #         T[2 * k + 1] = np.array([[0, t], [t, 0]])
    #
    #         # Matrice de diffusion d'interface
    #         b1 = gamma[k] / (f[k])
    #         b2 = gamma[k + 1] / (f[k + 1])
    #         T[2 * k + 2] = [[(b1 - b2) / (b1 + b2), (2 * b2) / (b1 + b2)],
    #                         [(2 * b1) / (b1 + b2), (b2 - b1) / (b1 + b2)]]
    #
    #     # Matrice de diffusion pour la dernière couche
    #     t = np.exp(1j * gamma[g - 1] * hauteur[g - 1])
    #     T[2 * g - 1] = [[0, t], [t, 0]]
    #
    #     H = np.zeros((2 * g - 1, 2, 2), dtype=complex)
    #     A = np.zeros((2 * g - 1, 2, 2), dtype=complex)
    #     H[0] = T[2 * g - 1]
    #     A[0] = T[0]
    #
    #     for j in range(2 * g - 2):
    #         A[j + 1] = self.cascade(A[j], T[j + 1])
    #         H[j + 1] = self.cascade(T[2 * g - 2 - j], H[j])
    #
    #     I = np.zeros((2 * g, 2, 2), dtype=complex)
    #     for j in range(len(T) - 1):
    #         I[j] = np.array([[A[j][1, 0], A[j][1, 1] * H[len(T) - j - 2][0, 1]],
    #                          [A[j][1, 0] * H[len(T) - j - 2][0, 0], H[len(T) - j - 2][0, 1]]] / (
    #                                 1 - A[j][1, 1] * H[len(T) - j - 2][0, 0]))
    #
    #     I[2 * g - 1] = np.array([[I[2 * g - 2][0, 0] * np.exp(1j * gamma(g) * hauteur(g)),
    #                               I[2 * g - 1][0, 1] * np.exp(1j * gamma(g) * hauteur(g))], [0, 0]])
    #
    #     w = 0
    #     poynting = np.zeros((1, 2 * g), dtype=complex)
    #
    #     if self.pol == 0: # TE
    #         for j in range(2 * g):
    #             poynting[j] = np.real(I[j][0, 0] + I[j][1, 0]) * np.conj((I[j][0, 0] - I[j][1, 0])* gamma[w] / Mu[w]) * \
    #                           Mu[0] / gamma[0]
    #             w = w + 1 - np.mod(j + 1, 2)
    #
    #     elif self.pol == 1: # TM
    #         for j in range(2 * g):
    #             poynting[j] = np.real(I[j][0, 0] - I[j][1, 0]) * np.conj((I[j][0, 0] + I[j][1, 0]) * gamma[w] / Mu[w]) * \
    #                           Mu[0] / gamma[0]
    #             w = w + 1 - np.mod(j + 1, 2)
    #
    #     tmp = np.abs(-np.diff(poynting))
    #     absorb = tmp[np.arange(0, 2 * g, 2)]
    #     return absorb
    #
    # def AbsAngular(self):
    #     """
    #     This program can compute the reflection and transmission coefficients
    #     as a function of the incidence angle, as well as the absorption in a
    #     specified layer.
    #     """
    #     # Number of the layer where the absorption has to be computed
    #     Layer = 2
    #     # Workgin wavelength
    #     lam = 500
    #     # Polarization - 0 means s or TE; 1 means p or TM.
    #     # polarization=0
    #     # Angular range in degrees here
    #     min = 0
    #     max = 89
    #     # Number of points
    #     Npoints = 200
    #     # [Epsilon,Mu,Type,hauteur,pol]=structure()
    #     Ab = np.zeros((Npoints, Type.size))
    #     theta = np.linspace(min, max, Npoints)
    #     r = np.zeros(Npoints, dtype=complex)
    #     t = np.zeros(Npoints, dtype=complex)
    #     for k in range(Npoints):
    #         tmp = theta[k] / 180 * pi
    #         [r[k], R, t[k], T] = coefficient(tmp, lam, pol)
    #     plt.figure(1)
    #     plt.subplot(311)
    #     plt.title("Modulus of the transmission coefficient")
    #     plt.plot(theta, abs(t))
    #     plt.ylabel("Absorption")
    #     plt.xlabel("Angle (degrees)")
    #     plt.subplot(312)
    #     plt.plot(theta, abs(r) ** 2)
    #     plt.ylabel("Coefficient")
    #     plt.xlabel("Angle")
    #     plt.title("Energy reflection coefficient")
    #     plt.subplot(313)
    #     plt.plot(theta, np.angle(r))
    #     plt.ylabel("Phase (in radians)")
    #     plt.xlabel("Angle in degree")
    #     plt.title("Phase of the reflection coefficient")
    #     plt.tight_layout()
    #     plt.show()
