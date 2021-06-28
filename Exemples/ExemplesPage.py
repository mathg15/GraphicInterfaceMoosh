import streamlit as st

from Tools.widget import *
from Exemples.Bragg import *
from Exemples.SPR import *

widget = sidebarWidget()


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
        st.sidebar.markdown(" ## Plasmons de surface")

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
