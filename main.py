"""
# Pokemon generator using GAN
Group 4: David Darmanger, Owen Gombas, Clément Brigliano
"""

# Imports
import streamlit as st
from notebooks.pokedex_generate import generate_pokedex
from notebooks.shuffle_generate import generate_shuffle
from PIL import Image

image = Image.open('notebooks/ressources/logos/title.png')
st.image(image)

st.markdown("# Welcome to our Pokemons generator using GANs")

st.markdown("## General")
st.markdown("- You can find the code on our [repo](https://gitlab-etu.ing.he-arc.ch/isc/2022-23/niveau-3/3281-projet-p3-hes-ete-id/pokemon)")
st.markdown("- You can find the documentation on [GitLab](https://gitlab-etu.ing.he-arc.ch/isc/2022-23/niveau-3/3281-projet-p3-hes-ete-id/pokemon/-/wikis/home)")

st.markdown("- Shuffle model also available on [Gradio](https://huggingface.co/spaces/clemsou/pokemon_generator)")

st.markdown("## Pokedex model")
st.pyplot(generate_pokedex())

st.markdown("## Shuffle model")
values = st.slider(
     'select the number of pokemon you want to generate',
     1, 30)
st.write('Values:', values)
st.pyplot(generate_shuffle(values))