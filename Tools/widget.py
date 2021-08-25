import streamlit as st


class sidebarWidget:

    def sliderPara(self):
        return st.slider("Nombre de miroirs", 1, 50, 20, 1)

    def checkPara(self):
        return st.radio("Polarisation", ("Polarisation TE", "Polarisation TM"))

    def angleInput1(self):
        return st.number_input("Angle d'incidence", 1, 90, 35, format=None, key=f"{1}")

    def angleInput2(self):
        return st.number_input("Angle d'incidence", 0, 90, 35, format=None, key=f"{2}")

    def angleInput3(self):
        return st.number_input("Angle d'incidence", 0, 90, 35, format=None, key=f"{3}")

    def lambdaInput1(self):
        return st.number_input("Longueur d'onde", 400, 800, 600, format=None, key=f"{21}")

    def lambdaInput2(self):
        return st.number_input("Longueur d'onde", 400, 800, 600, format=None, key=f"{222}")

    def lambdaInput3(self):
        return st.number_input("Longueur d'onde", 400, 800, 600, format=None, key=f"{23}")

    def lambdaInput4(self):
        return st.number_input("Longueur d'onde", 400, 800, 600, format=None, key=f"{24}")

    def beamPos(self):
        return st.slider("Position", 0.0, 1.0, 0.4, 0.1)

    def selectBox1(self):
        return st.selectbox("Choix du verre", ("BK7", "SiO2", "TiO2"))

    def struslider(self):
        return st.slider("Nombres de couches", 1, 10, 1, 1)

    def selecBoxGen1(self):
        return st.selectbox("Choix du matériau 1", ("Air", "Eau", "Bk7", "SiO2", "TiO2", "Au", "Cr","Ag","ZnO"))

    def hauteurGen1(self):
        return st.number_input("Epaisseur du matériau 1", 1, 20000, 400, 1, key=f"{1}")

    def selecBoxGen2(self):
        return st.selectbox("Choix du matériau 2", ("Air", "Eau", "Bk7", "SiO2", "TiO2", "Au", "Cr","Ag","ZnO"))

    def hauteurGen2(self):
        return st.number_input("Epaisseur du matériau 2", 1, 20000, 400, 1,  key=f"{2}")

    def selecBoxGen3(self):
        return st.selectbox("Choix du matériau 3", ("Air", "Eau", "Bk7", "SiO2", "TiO2", "Au", "Cr","Ag","ZnO"))

    def hauteurGen3(self):
        return st.number_input("Epaisseur du matériau 3", 1, 20000, 400, 1, key=f"{3}")

    def selecBoxGen4(self):
        return st.selectbox("Choix du matériau 4", ("Air", "Eau", "Bk7", "SiO2", "TiO2", "Au", "Cr","Ag","ZnO"), key=f"{4}")

    def hauteurGen4(self):
        return st.number_input("Epaisseur du matériau 4", 1, 20000, 400, 1, key=f"{4}")

    def selecBoxGen5(self):
        return st.selectbox("Choix du matériau 5", ("Air", "Eau", "Bk7", "SiO2", "TiO2", "Au", "Cr","Ag","ZnO"), key=f"{5}")

    def hauteurGen5(self):
        return st.number_input("Epaisseur du matériau 5", 1, 20000, 400, 1, key=f"{5}")

    def selecBoxGen6(self):
        return st.selectbox("Choix du matériau 6", ("Air", "Eau", "Bk7", "SiO2", "TiO2", "Au", "Cr","Ag","ZnO"), key=f"{6}")

    def hauteurGen6(self):
        return st.number_input("Epaisseur du matériau 6", 1, 20000, 400, 1, key=f"{6}")

    def selecBoxGen7(self):
        return st.selectbox("Choix du matériau 7", ("Air", "Eau", "Bk7", "SiO2", "TiO2", "Au", "Cr","Ag","ZnO"), key=f"{7}")
    
    def hauteurGen7(self):
        return st.number_input("Epaisseur du matériau 7", 1, 20000, 400, 1, key=f"{7}")

    def selecBoxGen8(self):
        return st.selectbox("Choix du matériau 8", ("Air", "Eau", "Bk7", "SiO2", "TiO2", "Au", "Cr","Ag","ZnO"), key=f"{8}")

    def hauteurGen8(self):
        return st.number_input("Epaisseur du matériau 8", 1, 20000, 400, 1, key=f"{8}")

    def selecBoxGen9(self):
        return st.selectbox("Choix du matériau 9", ("Air", "Eau", "Bk7", "SiO2", "TiO2", "Au", "Cr","Ag","ZnO"), key=f"{9}")

    def hauteurGen9(self):
        return st.number_input("Epaisseur du matériau 9", 1, 20000, 400, 1, key=f"{9}")
    
    def selecBoxGen10(self):
        return st.selectbox("Choix du matériau 10", ("Air", "Eau", "Bk7", "SiO2", "TiO2", "Au", "Cr","Ag","ZnO"), key=f"{10}")

    def hauteurGen10(self):
        return st.number_input("Epaisseur du matériau 10", 1, 20000, 400, 1, key=f"{10}")

    def angleCoefGen(self):
        return st.number_input("Angle d'incidence", 0, 90, 35, format=None, key=f"{4}")

    def lambdaCoefGen(self):
        return st.number_input("Longueur d'onde", 400, 800, 600, format=None, key=f"{5}")

    def lambAngGen(self):
        return st.number_input("Longueur d'onde", 400, 800, 600, format=None, key=f"{6}")

    def btnAngGen(self):
        return st.button("Afficher Angular")

    def thetaSpecGen(self):
        return st.number_input("Angle d'incidence", 0, 90, 35, format=None, key=f"{15}")

    def btnSpecGen(self):
        return st.button("Afficher Spectrum")

    def btnCoefGen(self):
        return st.button("Afficher les coefficients")

    def PermiSetGen(self):
        return st.number_input("Longueur d'onde", 400, 800, 600, format=None, key=f"{7}")

    def multiSelectPolaGen(self):
        return st.selectbox("Polarisation", ["Polarisation TE", "Polarisation TM"])

    def switchRefTraGen(self):
        return st.radio("", ("Reflexion", "Transmission"))

    def minWinAngGen(self):
        return st.number_input("Angle minimum", 0, 90, 0, format=None, key=f"{7}")

    def maxWinAngGen(self):
        return st.number_input("Angle maximum", 0, 90, 90, format=None, key=f"{8}")

    def lambBeamGen(self):
        return st.number_input("Longueur d'onde", 400, 800, 600, key=f"{14}")

    def angBeamGen(self):
        return st.number_input("Angle", 1.0, 90.0, 45.0, 0.1, key=f"{9}")

    def posBeamGen(self):
        return st.slider("Position", 0.0, 1.0, 0.5, 0.1, key=f"{17}")

    def btnBeamGen(self):
        return st.button("Afficher Beam")

    def polParaSPR(self):
        return st.selectbox("Polarisation", ["TE", "TM"])

    def angCoefSPR(self):
        return st.number_input("Angle", 1.0, 90.0, 45.0, 0.1, key=f"{10}")

    def angSpecSPR(self):
        return st.number_input("Angle", 1.0, 90.0, 45.0, 0.1, key=f"{11}")

    def angBeamSPR(self):
        return st.number_input("Angle", 1.0, 90.0, 45.0, 0.1, key=f"{12}")

    def lambCoefSPR(self):
        return st.number_input("Longueur d'onde", 400, 800, 600, key=f"{9}")

    def lambAngSPR(self):
        return st.number_input("Longueur d'onde", 400, 800, 600, key=f"{10}")

    def lambBeamSPR(self):
        return st.number_input("Longueur d'onde", 400, 800, 600, key=f"{11}")


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
        return st.number_input("Longueur d'onde", 400, 800, 600,key=f"{12}")

    def btnAbsGen(self):
        return st.button("Afficher l'absorption")

    def sliderLayerGen(self):
        return st.slider("Layer", 1, 5, 2)

