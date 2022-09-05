# Pokémon - HES d'été

## Groupe 4
+ Owen Gombas
+ Clément Brigliano
+ David Darmanger


## Structure
Il y a 2 modèles différents. 
+ Modèle 1 - **Pokedex** : dataset contentant des images entières de pokémons
+ Modèle 2 - **Shuffle** : dataset contenant des "icônes" de pokémons


### Fichiers
+ La page de présentation streamlit se trouvent dans `main.py`
+ Le modèle 1 entraîné se trouve dans `generator_pokedex.h5`
+ Le modèle 2 entraîné se trouve dans `generator_shuffle.h5`

+ Les notebooks des différentes model se trouvent dans : `/notebooks/`
    + `../pokedex_generate.ipynb` : notebook de génération de pokedex (modèle 1)
    + `../pokedex_generate.py` : script de génération de pokedex (modèle 1)
    + `../pokedex_train.ipynb` : notebook d'entrainement du modèle 1
    + `../shuffle_generation.ipynb` : notebook de génération de shuffle (modèle 2)
    + `../shuffle_generation.py` : script de génération de shuffle (modèle 2)

+ Les différentes Classes utilitaires des modèles se trouvent dans : `/notebooks/tools`
+ Les différents images générées, logs et logos utilisés se trouvent dans : `/notebooks/ressources`
+ Les fichiers nécessaires pour le déploiement gradio se trouvent dans `/gradio`

# Références
- https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html 
- https://www.tensorflow.org/tutorials/generative/dcgan
- https://www.youtube.com/watch?v=JB8T_zN7ZC0&t=2627s
