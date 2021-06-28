import numpy as np
import matplotlib.pyplot as plt
from scipy import special
import streamlit as st
import matplotlib.image as im
import matplotlib.colors as mcolors

# Fichiers annexes

from MooshGen import *
from mat import *
from widget import *
from Pages.homepage import *
from Pages.docs import *
from Exemples.Bragg import *
from Exemples.SPR import *

# Haut de la page

st.set_page_config(page_title="Moosh", page_icon="./Images/logo_moosh.jpg")


def genmoosh():
    st.write("MooshGen")
    st.write("Structure")
    with st.sidebar.beta_expander('Structure'):
        ####
        st.markdown(" ## Sélection du nombre de couche de la structure et de la polarisation")
        strucSlider = widget.struslider()
        pol = widget.multiSelectPolaGen()
        st.markdown(" ## Choix des matériaux de la structure")
        gen = mooshGen(strucSlider, pol)
        # gen.structure(permSet)

    with st.sidebar.beta_expander('Coefficients'):
        st.markdown("## Coefficients de réflexion et de transmittance de la structure")
        st.write("Définition de la longeur d'onde")
        coefLambGen = widget.lambdaCoefGen()
        st.write("Définition de l'angle")
        coefAngGen = widget.angleCoefGen()

        btnCoefGen = widget.btnCoefGen()

    if btnCoefGen == 1:
        gen.affichageCoef(coefAngGen, coefLambGen)

    with st.sidebar.beta_expander('Angular'):
        st.markdown("## Coefficient de réflexion en fonction de l'angle")
        st.write("Définition d'une longeur d'onde constante")
        angLambGen = widget.lambAngGen()
        st.write("Intervalle angulaire")
        angMaxAnGen = widget.maxWinAngGen()
        angMinAnGen = widget.minWinAngGen()

        switchRT = widget.switchRefTraGen()
        btnAngGen = widget.btnAngGen()

    if btnAngGen == 1:
        gen.angular(angLambGen, switchRT, angMinAnGen, angMaxAnGen)

    # with st.sidebar.beta_expander("Absorption"):
    #     st.markdown("## Absorption")
    #     st.write("Définition d'une longeur d'onde constante")
    #     absLambGen = widget.absLambGen()
    #     absLayerGen = widget.sliderLayerGen()
    #     btnAbs = widget.btnAbsGen()
    #
    # if btnAbs == 1:
    #     gen.absorptionAngular(absLambGen,absLayerGen)

    with st.sidebar.beta_expander('Spectrum'):
        st.markdown("## Coefficient de réflexion en fonction de la longueur d'onde")
        st.write("Définition d'un angle constant")
        specAngGen = widget.thetaSpecGen()
        btnSpecGen = widget.btnSpecGen()

    if btnSpecGen == 1:
        gen.spectrum(specAngGen)

    with st.sidebar.beta_expander('Beam'):
        st.markdown("## Modélisation du faisceau lumineux")
        beamAngGen = widget.angBeamGen()
        beamLambGen = widget.lambBeamGen()
        beamPosGen = widget.posBeamGen()
        btnBeamGen = widget.btnBeamGen()

    if btnBeamGen == 1:
        gen.beam(beamLambGen, beamAngGen, beamPosGen)


def exmoosh():
    sideBarExp = st.sidebar.radio("Choix de l'expérience", ('Miroir de Bragg', 'Plasmon de surface', 'Photovoltaïque'))
    if sideBarExp == 'Miroir de Bragg':
        st.sidebar.markdown(" ## Miroir de Bragg")
        with st.sidebar.beta_expander(" Paramètres"):
            ####
            # st.markdown(" ## Paramètres")
            mirpara = widget.sliderPara()
            polparab = widget.checkPara()

        with st.sidebar.beta_expander("Coefficients"):
            # st.markdown(" ## Coefficients")
            coefAng = widget.lambdaInput1()
            coefLamb = widget.angleInput1()
            btnCoef = st.button("Afficher les coefficients")
            ####
        with st.sidebar.beta_expander("Angular"):
            # st.markdown(" ## Angular")
            angLamb = widget.lambdaInput2()
            btnAng = st.button("Afficher Angular")

        with st.sidebar.beta_expander("Spectrum"):
            # st.markdown(" ## Spectrum")
            specAngle = widget.angleInput2()
            btnSpec = st.button("Afficher Spectrum")

        with st.sidebar.beta_expander("Beam"):
            # st.markdown(" ## Beam")
            bPos = widget.beamPos()
            beamLamb = widget.lambdaInput3()
            beamAng = widget.angleInput3()
            btnBeam = st.button("Afficher Beam")

        Bragg_ = Bragg(mirpara, polparab)

        if btnCoef == 1:
            Bragg_.affichageCoef(coefAng, coefLamb)

        if btnAng == 1:
            Bragg_.angular(angLamb)

        if btnSpec == 1:
            Bragg_.spectrum(specAngle)

        if btnBeam == 1:
            Bragg_.beam(beamLamb, beamAng, bPos)

    elif sideBarExp == 'Plasmon de surface':
        st.text("SPR")
        st.text("")
        with st.sidebar.beta_expander(" Paramètres"):
            st.markdown(" ## Paramètres")
            polSpr = widget.polParaSPR()

        with st.sidebar.beta_expander("Coefficients"):
            # st.markdown(" ## Coefficients")
            coefAng = widget.angCoefSPR()
            coefLamb = widget.lambCoefSPR()
            btnCoef = widget.btnCoefSPR()
            ####
        with st.sidebar.beta_expander("Angular"):
            # st.markdown(" ## Angular")
            angLamb = widget.lambAngSPR()
            btnAng = st.button("Afficher Angular")

        with st.sidebar.beta_expander("Spectrum"):
            # st.markdown(" ## Spectrum")
            specAngle = widget.angSpecSPR()
            btnSpec = st.button("Afficher Spectrum")

        with st.sidebar.beta_expander("Beam"):
            # st.markdown(" ## Beam")
            bPos = widget.posBeamSPR()
            beamLamb = widget.lambBeamSPR()
            beamAng = widget.angBeamSPR()
            btnBeam = st.button("Afficher Beam")

        _SPR = SPR(polSpr)

        if btnCoef == 1:
            _SPR.affichageCoef(coefAng, coefLamb)

        if btnAng == 1:
            _SPR.angular(angLamb)

        if btnSpec == 1:
            _SPR.spectrum(specAngle)

        if btnBeam == 1:
            _SPR.beam(beamLamb, beamAng, bPos)

    elif sideBarExp == 'Photovoltaïque':
        st.text("Work in progress")


widget = sidebarWidget()

st.sidebar.image("./Images/logo_moosh.jpg")
st.sidebar.title("Navigation")
st.sidebar.write('')

side_menu_navigation = st.sidebar.radio('', ('Homepage', 'Moosh', 'Exemples', 'Documentation'))
if side_menu_navigation == 'Homepage':
    homepage()
elif side_menu_navigation == 'Moosh':
    genmoosh()
elif side_menu_navigation == 'Exemples':
    exmoosh()
elif side_menu_navigation == 'Documentation':
    documentation()
