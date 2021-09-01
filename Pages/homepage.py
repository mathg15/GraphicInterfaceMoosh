import streamlit as st

def homepage():
    st.title("Moosh")
    st.header("Le couteau suisse numérique pour l'optique")
    st.write("")
    st.write("Le logiciel Moosh a pour objectif de modéliser un faisceau lumineux à travers un structure multicouche.")
    st.header("Examples de réalisations : ")
    st.markdown("## Coefficient de réflexion en fonction de l'angle pour une structure multicouche contenant du Chrome")
    st.image("./Images/reflexionCrTM.png")
    st.markdown("## Faisceau lumineux dans un lame de verre SiO2")
    st.image("./Images/beam")
