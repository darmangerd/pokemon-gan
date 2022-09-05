"""
# Pokemon generator using GAN
Group 4: David Darmanger, Owen Gombas, Clément Brigliano
"""

# Imports
import streamlit as st
from notebooks.pokedex_generate import generate_pokedex
from notebooks.shuffle_generate import generate_shuffle



st.markdown("# Welcome to our Pokemon generator using GAN")

st.markdown("## General")
st.markdown("- You can find the code on our [repo](https://gitlab-etu.ing.he-arc.ch/isc/2022-23/niveau-3/3281-projet-p3-hes-ete-id/pokemon)")
st.markdown("- You can find the documentation on [GitLab](https://gitlab-etu.ing.he-arc.ch/isc/2022-23/niveau-3/3281-projet-p3-hes-ete-id/pokemon/-/wikis/home)")

st.markdown("## Pokedex model")
st.pyplot(generate_pokedex())

st.markdown("## Shuffle model")
st.pyplot(generate_shuffle(6))
