import streamlit as st

from Tools.widget import *
from Main.MooshGen import *

widget = sidebarWidget()


def genmoosh():


    with st.sidebar.expander('Structure'):
        ####
        st.markdown(" ## Selection of the number of layers of the structure and the polarisation.")
        strucSlider = widget.struslider()
        pol = widget.multiSelectPolaGen()
        st.markdown(" ## Choice of materials for the structure")
        gen = mooshGen(strucSlider, pol)
        # gen.structure(permSet)

    with st.sidebar.expander('Coefficients'):
        st.markdown("## Reflection and transmittance coefficients of the structure")
        st.write("Definition of the wavelength")
        coefLambGen = widget.lambdaCoefGen()
        st.write("Definition of the angle")
        coefAngGen = widget.angleCoefGen()

        btnCoefGenEN = widget.btnCoefGenEN()

    

    if btnCoefGenEN == 1:
        gen.affichageCoef(coefAngGen, coefLambGen)

    with st.sidebar.expander('Angular'):
        st.markdown("## Reflection coefficient as a function of angle")
        st.write("Definition of a constant wavelength")
        angLambGen = widget.lambAngGen()
        st.write("Angular interval")
        angMaxAnGen = widget.maxWinAngGen()
        angMinAnGen = widget.minWinAngGen()

        switchRT = widget.switchRefTraGen()
        btnAngGenEN = widget.btnAngGenEN()


    if btnAngGenEN == 1:
        gen.angular(angLambGen, switchRT, angMinAnGen, angMaxAnGen)

    # with st.sidebar.expander("Absorption"):
    #     st.markdown("## Absorption")
    #     st.write("Définition d'une longeur d'onde constante")
    #     absLambGen = widget.absLambGen()
    #     absLayerGen = widget.sliderLayerGen()
    #     btnAbs = widget.btnAbsGen()
    #
    # if btnAbs == 1:
    #     gen.absorptionAngular(absLambGen,absLayerGen)

    with st.sidebar.expander('Spectrum'):
        st.markdown("## Reflection coefficient as a function of wavelength")
        st.write("Definition of a constant angle")
        specAngGen = widget.thetaSpecGen()
        btnSpecGenEN = widget.btnSpecGenEN()


    if btnSpecGenEN == 1:
        gen.spectrum(specAngGen)

    
    with st.sidebar.expander('Beam'):
        st.markdown("## Modeling the light beam")
        beamAngGen = widget.angBeamGen()
        beamLambGen = widget.lambBeamGen()
        beamPosGen = widget.posBeamGen()
        btnBeamGenEN = widget.btnBeamGenEN()

    if btnBeamGenEN == 1:
        gen.beam(beamLambGen, beamAngGen, beamPosGen)
