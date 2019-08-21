# https://colab.research.google.com/drive/1RUaVUqCvyojwoMglp6cFoLDnCfLHBZtB#scrollTo=PVYrjvgE_8AU

import helpers
import os
import pickle
import PIL.Image
import numpy as np
import dnnlib
import dnnlib.tflib as tflib
import config
from encoder.generator_model import Generator
import matplotlib.pyplot as plt
import runway




@runway.setup(options={'checkpoint': runway.file(extension='.pkl')})
def setup(opts):
	# load direction
	global direction
	age_direction = np.load('ffhq_dataset/latent_directions/age.npy')
	direction = age_direction
	# load latent representation
	global latent_vector
	r1 = 'latent_representations/j_01.npy'
	latent_vector = np.load(r1)
	# load checkpoint
    tflib.init_tf()
    with open(opts['checkpoint'], 'rb') as file:
        G, D, Gs = pickle.load(file)
    #age_direction = np.load('ffhq_dataset/latent_directions/age.npy') 
    # generator
    global generator   
    generator = Generator(Gs, batch_size=1, randomize_noise=False)
    return Gs



generate_inputs = {
    'age': runway.number(min=-6, max=6, default=6, step=0.1)
}

@runway.command('generate', inputs=generate_inputs, outputs={'image': runway.image})
def move_and_show(latent_vector, direction, inputs):
	coeffs = inputs['age']
    fig,ax = plt.subplots(1, len(coeffs), figsize=(15, 10), dpi=80)
    for i, coeff in enumerate(coeffs):
        new_latent_vector = latent_vector.copy()
        new_latent_vector[:8] = (latent_vector + coeff*direction)[:8]
        ax[i].imshow(generate_image(generator, new_latent_vector))
        ax[i].set_title('Coeff: %0.1f' % coeff)
    [x.axis('off') for x in ax]
    #plt.show()
    output = fig2data(plt)
    return {'image': output}

if __name__ == '__main__':
    runway.run()