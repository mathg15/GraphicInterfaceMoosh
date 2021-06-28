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
