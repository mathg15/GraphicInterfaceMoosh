import streamlit as st

def documentation():
    st.title('Documentation')
    choix = st.radio("Liste des documents",("Utilisation de l'application","Théorie","Bibliographie"))
    if choix == "Utilisation de l'application":
        st.header("Utilisation de l'application")
    elif choix == "Théorie":
        st.header("Théorie")
    elif choix == "Bibliographie"
        st.header("Bibliographie")
    st.write("Utilisation de Moosh")
