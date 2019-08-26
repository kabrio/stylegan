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


@runway.setup
def setup(opts):
	global Gs
	tflib.init_tf()
	url = 'https://drive.google.com/uc?id=1MEGjdvVpUsu1jB4zrXZN7Y4kBBOzizDQ' # karras2019stylegan-ffhq-1024x1024.pkl
	with dnnlib.util.open_url(url, cache_dir=config.cache_dir) as f:
		_G, _D, Gs = pickle.load(f)
		# _G = Instantaneous snapshot of the generator. Mainly useful for resuming a previous training run.
		# _D = Instantaneous snapshot of the discriminator. Mainly useful for resuming a previous training run.
		# Gs = Long-term average of the generator. Yields higher-quality results than the instantaneous snapshot.
	Gs.print_layers()
	return Gs

def generate_image(generator, latent_vector):
	latent_vector = latent_vector.reshape((1, 18, 512))
	generator.set_dlatents(latent_vector)
	img_array = generator.generate_images()[0]
	img = PIL.Image.fromarray(img_array, 'RGB')
	return img.resize((512, 512))   

generate_inputs = {
	'age': runway.number(min=-6, max=6, default=6, step=0.1)
}

generate_outputs = {
	'image': runway.image(width=512, height=512)
}

@runway.command('generat3', inputs=generate_inputs, outputs=generate_outputs)
def move_and_show(model, args):
	# load direction
	age_direction = np.load('ffhq_dataset/latent_directions/age.npy')
	direction = age_direction
	# load latent representation
	r1 = 'latent_representations/j_01.npy'
	latent_vector = np.load(r1)
	# generator
	generator = Generator(model, batch_size=1, randomize_noise=False)
	coeff = inputs['age']
	fig,ax = plt.subplots(1, 1, figsize=(15, 10), dpi=80)
	new_latent_vector = latent_vector.copy()
	new_latent_vector[:8] = (latent_vector + coeff*direction)[:8]
	image = (generate_image(generator, new_latent_vector))
	#ax[i].set_title('Coeff: %0.1f' % coeff)
	#plt.show()
	return {'image': image}

if __name__ == '__main__':
	runway.run(debug=True)