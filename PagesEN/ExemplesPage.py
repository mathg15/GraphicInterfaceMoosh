import streamlit as st

from Tools.widget import *
from Exemples.Bragg import *
from Exemples.SPR import *

widget = sidebarWidget()


def exmoosh():
    sideBarExp = st.sidebar.radio("Choice of the experiment", ('Bragg mirror', 'Surface Plasmon', 'Photovoltaic'))
    if sideBarExp == 'Bragg mirror':
        st.sidebar.markdown(" ## Bragg mirror")
        with st.sidebar.expander("Settings"):
            ####
            # st.markdown(" ## Paramètres")
            mirpara = widget.sliderPara()
            polparab = widget.checkPara()

        with st.sidebar.expander("Coefficients"):
            # st.markdown(" ## Coefficients")
            coefAng = widget.lambdaInput1()
            coefLamb = widget.angleInput1()
            btnCoef = st.button("Show coefficients")
            ####
        with st.sidebar.expander("Angular"):
            # st.markdown(" ## Angular")
            angLamb = widget.lambdaInput2()
            btnAng = st.button("Show Angular")

        with st.sidebar.expander("Spectrum"):
            # st.markdown(" ## Spectrum")
            specAngle = widget.angleInput2()
            btnSpec = st.button("Show Spectrum")

        with st.sidebar.expander("Beam"):
            # st.markdown(" ## Beam")
            bPos = widget.beamPos()
            beamLamb = widget.lambdaInput3()
            beamAng = widget.angleInput3()
            btnBeam = st.button("Show Beam")

        Bragg_ = Bragg(mirpara, polparab)

        if btnCoef == 1:
            Bragg_.affichageCoef(coefAng, coefLamb)

        if btnAng == 1:
            Bragg_.angular(angLamb)

        if btnSpec == 1:
            Bragg_.spectrum(specAngle)

        if btnBeam == 1:
            Bragg_.beam(beamLamb, beamAng, bPos)

    elif sideBarExp == 'Surface Plasmon':
        st.sidebar.markdown(" ## Surface Plasmon")

        with st.sidebar.expander("Settings"):
            # st.markdown(" ## Settings")
            polSpr = widget.polParaSPR()

        with st.sidebar.expander("Coefficients"):
            # st.markdown(" ## Coefficients")
            coefAng = widget.angCoefSPR()
            coefLamb = widget.lambCoefSPR()
            btnCoef = widget.btnCoefSPR()
            ####
        with st.sidebar.expander("Angular"):
            # st.markdown(" ## Angular")
            angLamb = widget.lambAngSPR()
            btnAng = st.button("Show Angular")

        with st.sidebar.expander("Spectrum"):
            # st.markdown(" ## Spectrum")
            specAngle = widget.angSpecSPR()
            btnSpec = st.button("Show Spectrum")

        with st.sidebar.expander("Beam"):
            # st.markdown(" ## Beam")
            bPos = widget.posBeamSPR()
            beamLamb = widget.lambBeamSPR()
            beamAng = widget.angBeamSPR()
            btnBeam = st.button("Show Beam")

        _SPR = SPR(polSpr)

        if btnCoef == 1:
            _SPR.affichageCoef(coefAng, coefLamb)

        if btnAng == 1:
            _SPR.angular(angLamb)

        if btnSpec == 1:
            _SPR.spectrum(specAngle)

        if btnBeam == 1:
            _SPR.beam(beamLamb, beamAng, bPos)

    elif sideBarExp == 'Photovoltaic':
        st.text("Work in progress")
