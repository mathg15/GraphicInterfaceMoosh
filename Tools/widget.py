import streamlit as st


class sidebarWidget:

    def sliderPara(self):
        return st.slider("Nombre de miroirs / Number of mirrors", 1, 50, 20, 1)

    def checkPara(self):
        return st.radio("Polarisation", ("TE", "TM"))

    def angleInput1(self):
        return st.number_input("Angle d'incidence / Incidence angle", 1, 90, 35, format=None, key=f"{1}")

    def angleInput2(self):
        return st.number_input("Angle d'incidence / Incidence angle", 0, 90, 35, format=None, key=f"{2}")

    def angleInput3(self):
        return st.number_input("Angle d'incidence / Incidence angle", 0, 90, 35, format=None, key=f"{3}")

    def lambdaInput1(self):
        return st.number_input("Longueur d'onde / Wavelength", 400, 800, 600, format=None, key=f"{21}")

    def lambdaInput2(self):
        return st.number_input("Longueur d'onde / Wavelength", 400, 800, 600, format=None, key=f"{222}")

    def lambdaInput3(self):
        return st.number_input("Longueur d'onde / Wavelength", 400, 800, 600, format=None, key=f"{23}")

    def lambdaInput4(self):
        return st.number_input("Longueur d'onde / Wavelength", 400, 800, 600, format=None, key=f"{24}")

    def beamPos(self):
        return st.slider("Position", 0.0, 1.0, 0.4, 0.1)

    def struslider(self):
        return st.slider("Nombres de couches / Number of layers", 1, 10, 1, 1)

    def selecBoxGen1(self):
        return st.selectbox("Choix du matériau 1 / Choice of the material 1", ("Air", "Eau", "Bk7", "SiO2", "TiO2", "Au", "Cr","Ag","ZnO"))

    def hauteurGen1(self):
        return st.number_input("Epaisseur du matériau 1 / Thickness of the material 1", 1, 20000, 400, 1, key=f"{1}")

    def selecBoxGen2(self):
        return st.selectbox("Choix du matériau 2 / Choice of the material 2 ", ("Air", "Eau", "Bk7", "SiO2", "TiO2", "Au", "Cr","Ag","ZnO"))

    def hauteurGen2(self):
        return st.number_input("Epaisseur du matériau 2 / Thickness of the material 2", 1, 20000, 400, 1,  key=f"{2}")

    def selecBoxGen3(self):
        return st.selectbox("Choix du matériau 3 / Choice of the material 3", ("Air", "Eau", "Bk7", "SiO2", "TiO2", "Au", "Cr","Ag","ZnO"))

    def hauteurGen3(self):
        return st.number_input("Epaisseur du matériau 3 / Thickness of the material 3", 1, 20000, 400, 1, key=f"{3}")

    def selecBoxGen4(self):
        return st.selectbox("Choix du matériau 4 / Choice of the material 4", ("Air", "Eau", "Bk7", "SiO2", "TiO2", "Au", "Cr","Ag","ZnO"), key=f"{144}")

    def hauteurGen4(self):
        return st.number_input("Epaisseur du matériau 4 / Thickness of the material 4", 1, 20000, 400, 1, key=f"{104}")

    def selecBoxGen5(self):
        return st.selectbox("Choix du matériau 5 / Choice of the material 5", ("Air", "Eau", "Bk7", "SiO2", "TiO2", "Au", "Cr","Ag","ZnO"), key=f"{105}")

    def hauteurGen5(self):
        return st.number_input("Epaisseur du matériau 5 / Thickness of the material 5", 1, 20000, 400, 1, key=f"{1555}")

    def selecBoxGen6(self):
        return st.selectbox("Choix du matériau 6 / Choice of the material 6", ("Air", "Eau", "Bk7", "SiO2", "TiO2", "Au", "Cr","Ag","ZnO"), key=f"{106}")

    def hauteurGen6(self):
        return st.number_input("Epaisseur du matériau 6 / Thickness of the material 6", 1, 20000, 400, 1, key=f"{166}")

    def selecBoxGen7(self):
        return st.selectbox("Choix du matériau 7 / Choice of the material 7", ("Air", "Eau", "Bk7", "SiO2", "TiO2", "Au", "Cr","Ag","ZnO"), key=f"{107}")
    
    def hauteurGen7(self):
        return st.number_input("Epaisseur du matériau 7 / Thickness of the material 7", 1, 20000, 400, 1, key=f"{177}")

    def selecBoxGen8(self):
        return st.selectbox("Choix du matériau 8 / Choice of the material 8", ("Air", "Eau", "Bk7", "SiO2", "TiO2", "Au", "Cr","Ag","ZnO"), key=f"{108}")

    def hauteurGen8(self):
        return st.number_input("Epaisseur du matériau 8 / Thickness of the material 8", 1, 20000, 400, 1, key=f"{188}")

    def selecBoxGen9(self):
        return st.selectbox("Choix du matériau 9 / Choice of the material 9", ("Air", "Eau", "Bk7", "SiO2", "TiO2", "Au", "Cr","Ag","ZnO"), key=f"{109}")

    def hauteurGen9(self):
        return st.number_input("Epaisseur du matériau 9 / Thickness of the material 9", 1, 20000, 400, 1, key=f"{199}")
    
    def selecBoxGen10(self):
        return st.selectbox("Choix du matériau 10 / Choice of the material 10", ("Air", "Eau", "Bk7", "SiO2", "TiO2", "Au", "Cr","Ag","ZnO"), key=f"{110}")

    def hauteurGen10(self):
        return st.number_input("Epaisseur du matériau 10 / Thickness of the material 10", 1, 20000, 400, 1, key=f"{111}")

    def angleCoefGen(self):
        return st.number_input("Angle d'incidence", 0, 90, 35, format=None, key=f"{4}")

    def lambdaCoefGen(self):
        return st.number_input("Longueur d'onde / Wavelength", 400, 800, 600, format=None, key=f"{5}")

    def lambAngGen(self):
        return st.number_input("Longueur d'onde / Wavelength", 400, 800, 600, format=None, key=f"{6}")

    def btnAngGen(self):
        return st.button("Afficher Angular")

    def thetaSpecGen(self):
        return st.number_input("Angle d'incidence / Incidence angle", 0, 90, 35, format=None, key=f"{15}")

    def btnSpecGen(self):
        return st.button("Afficher Spectrum")

    def btnCoefGen(self):
        return st.button("Afficher les coefficients")

    def PermiSetGen(self):
        return st.number_input("Longueur d'onde / Wavelength", 400, 800, 600, format=None, key=f"{7}")

    def multiSelectPolaGen(self):
        return st.radio("Polarisation", ["TE", "TM"],key=f"{566}")

    def switchRefTraGen(self):
        return st.radio("", ("Reflexion", "Transmission"))

    def minWinAngGen(self):
        return st.number_input("Angle minimum", 0, 90, 0, format=None, key=f"{7}")

    def maxWinAngGen(self):
        return st.number_input("Angle maximum", 0, 90, 90, format=None, key=f"{8}")

    def lambBeamGen(self):
        return st.number_input("Longueur d'onde / Wavelength", 400, 800, 600, key=f"{14}")

    def angBeamGen(self):
        return st.number_input("Angle", 1.0, 90.0, 45.0, 0.1, key=f"{9}")

    def posBeamGen(self):
        return st.slider("Position", 0.0, 1.0, 0.5, 0.1, key=f"{17}")

    def btnBeamGen(self):
        return st.button("Afficher Beam")

    def polParaSPR(self):
        return st.selectbox("Polarisation", ["TE", "TM"])

    def angCoefSPR(self):
        return st.number_input("Angle", 1.0, 90.0, 45.0, 0.1, key=f"{31}")

    def angSpecSPR(self):
        return st.number_input("Angle", 1.0, 90.0, 45.0, 0.1, key=f"{32}")

    def angBeamSPR(self):
        return st.number_input("Angle", 1.0, 90.0, 45.0, 0.1, key=f"{33}")

    def lambCoefSPR(self):
        return st.number_input("Longueur d'onde / Wavelength", 400, 800, 600, key=f"{34}")

    def lambAngSPR(self):
        return st.number_input("Longueur d'onde / Wavelength", 400, 800, 600, key=f"{35}")

    def lambBeamSPR(self):
        return st.number_input("Longueur d'onde / Wavelength", 400, 800, 600, key=f"{36}")


    def btnAngSPR(self):
        return st.button("Afficher Angular")

    def btnSpecSPR(self):
        return st.button("Afficher Spectrum")

    def btnCoefSPR(self):
        return st.button("Afficher les coefficients")

    def btnBeamSPR(self):
        return st.button("Afficher Beam")

    def posBeamSPR(self):
        return st.slider("Position", 0.0, 1.0, 0.4, 0.1, key=f"{3}")

    def absLambGen(self):
        return st.number_input("Longueur d'onde", 400, 800, 600,key=f"{37}")

    def btnAbsGen(self):
        return st.button("Afficher l'absorption")

    def sliderLayerGen(self):
        return st.slider("Layer", 1, 5, 2)

