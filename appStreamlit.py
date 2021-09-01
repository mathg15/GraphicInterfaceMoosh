import numpy as np
import matplotlib.pyplot as plt
from scipy import special
import streamlit as st
import matplotlib.image as im
import matplotlib.colors as mcolors

# Fichiers annexes

from Tools.mat import *
from Tools.widget import *
from Main.MooshGen import *
from Pages.homepage import *
from Pages.MooshGenPage import *
from Pages.ExemplesPage import *
from Pages.docs import *
from Exemples.Bragg import *
from Exemples.SPR import *

# Haut de la page

st.set_page_config(page_title="Moosh", page_icon="./Images/logo_moosh.jpg")

widget = sidebarWidget()

langSelec = st.sidebar.selectbox("", ("France", "English"))

if langSelec == 'France':

    from Pages.homepage import *
    from Pages.MooshGenPage import *
    from Pages.ExemplesPage import *
    from Pages.docs import *
    from Pages.maj import maj

    ex = "Exemples"
    up = "Mise à jour"

elif langSelec == 'English':

    from PagesEN.homepage import *
    from PagesEN.MooshGenPage import *
    from PagesEN.ExemplesPage import *
    from PagesEN.docs import *
    from PagesEN.maj import maj

    ex = "Examples"
    up = "Update"


st.sidebar.image("./Images/logo_moosh.jpg")
st.sidebar.title("Navigation")
st.sidebar.write('')

side_menu_navigation = st.sidebar.radio('', ('Homepage', 'Moosh', 'Exemples', 'Documentation'))
if side_menu_navigation == 'Homepage':
    homepage()
elif side_menu_navigation == 'Moosh':
    genmoosh()
elif side_menu_navigation == 'Examples':
    exmoosh()
elif side_menu_navigation == 'Documentation':
    documentation()
elif side_menu_navigation == 'Updates':
    maj()
