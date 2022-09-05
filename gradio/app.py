import gradio as gr
import numpy as np
from torchvision.utils import save_image
import torch.nn as nn
import torch
import torchvision.datasets as datasets
import torchvision.transforms as T
from torch.utils.data import DataLoader, ConcatDataset
import torchvision.utils as vutils
import random

LATENT_VECTOR_DIM = 8 # latent vector dimension

class Generator_128(nn.Module):
    def __init__(self, GPU_COUNT):
        super(Generator_128, self).__init__()
        self.GPU_COUNT = GPU_COUNT
        self.main = nn.Sequential(
            # LATENT_VECTOR_DIM x 1 x 1
            nn.ConvTranspose2d(LATENT_VECTOR_DIM, 1024, 4, 1, 0, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(True),
            nn.ConvTranspose2d(1024, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128,64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64,3, 4, 2, 1, bias=False),
            nn.Tanh()
            # 128 x 128 x 3
        )
    def forward(self, input):
        return self.main(input)



trained_gen = Generator_128(0)
trained_gen.load_state_dict(torch.load("generator_epoch_1300v1-5.h5",map_location=torch.device('cpu')))


def predict(seed, pokemon_count):
    torch.manual_seed(seed)
    z = torch.randn(pokemon_count, LATENT_VECTOR_DIM, 1, 1)
    pokemon = trained_gen(z)
    save_image(pokemon, "pokemon.png", normalize=True)
    return 'pokemon.png'

gr.Interface(
    predict,
    inputs=[
        gr.Slider(0, 1000, label='Seed', default=42),
        gr.Slider(1, 8, label='Number of pokemon', step=1, default=10),
    ],
    outputs="image",
).launch(share=True)



