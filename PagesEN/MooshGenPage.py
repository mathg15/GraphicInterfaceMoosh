import streamlit as st

from Tools.widget import *
from Main.MooshGen import *

widget = sidebarWidget()


def genmoosh():

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
