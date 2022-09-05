# Imports
from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML
from .gan.constants import *
from .gan.discriminator import Discriminator
from .gan.generator import Generator


def generate_pokedex():

    # Decide which device we want to run on
    device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

    # Create the generator
    netG = Generator(ngpu).to(device)

    # Handle multi-gpu if desired
    if (device.type == 'cuda') and (ngpu > 1):
        netG = nn.DataParallel(netG, list(range(ngpu)))

    # Apply the weights_init function to randomly initialize all weights
    #  to mean=0, stdev=0.02.
    netG.load_state_dict(torch.load('generator', map_location=torch.device('cpu')))
    netG.eval()

    # Set random seed for reproducibility
    seed = 2
    seed = random.randint(1, 10000) # use if you want new results

    fixed_noise = torch.randn(64, nz, 1, 1, device=device)

    random.seed(seed)
    torch.manual_seed(seed)
    print("Seed: ", seed)

    generated_images = netG(fixed_noise).detach().cpu()
    im = vutils.make_grid(generated_images, padding=0, normalize=True)
    fig = plt.figure(figsize=(20, 20))
    fig.set_facecolor('white')
    plt.imshow(np.transpose(im, (1, 2, 0)))
    plt.axis('off')
    return plt

