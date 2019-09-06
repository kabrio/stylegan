# https://colab.research.google.com/drive/1RUaVUqCvyojwoMglp6cFoLDnCfLHBZtB#scrollTo=PVYrjvgE_8AU

import helpers
import os
import argparse
import pickle
import PIL.Image
import numpy as np
import dnnlib
import dnnlib.tflib as tflib
import config
from encoder.generator_model import Generator
from encoder.perceptual_model import PerceptualModel
import matplotlib.pyplot as plt
from tqdm import tqdm
import runway

latent_vector_1=0
latent_vector_2=0
prevIterations = -1
generated_dlatents = 0

setup_options = {
	'checkpoint': runway.file(extension='.pkl'), 
	'people_vector': runway.file(extension='.npy'),
	'people_vector2': runway.file(extension='.npy')
}

@runway.setup(options=setup_options)
def setup(opts):
	tflib.init_tf()
	model = opts['checkpoint']
	print("open model %s" % model)
	with open(model, 'rb') as file:
		G, D, Gs = pickle.load(file)
	Gs.print_layers()
	# load latent representation
	p1 = inputs['people_vector']
	global latent_vector_1
	latent_vector_1 = np.load(p1)
	p2 = inputs['people_vector2']
	global latent_vector_2
	latent_vector_2 = np.load(p2)
	global generator
	generator = Generator(Gs, batch_size=1, randomize_noise=False)
	return generator

# def setup(opts):
# 	tflib.init_tf()
# 	url = 'https://drive.google.com/uc?id=1MEGjdvVpUsu1jB4zrXZN7Y4kBBOzizDQ' # karras2019stylegan-ffhq-1024x1024.pkl
# 	with dnnlib.util.open_url(url, cache_dir=config.cache_dir) as f:
# 		_G, _D, Gs = pickle.load(f)
# 		# _G = Instantaneous snapshot of the generator. Mainly useful for resuming a previous training run.
# 		# _D = Instantaneous snapshot of the discriminator. Mainly useful for resuming a previous training run.
# 		# Gs = Long-term average of the generator. Yields higher-quality results than the instantaneous snapshot.
# 	Gs.print_layers()
# 	global generator
# 	generator = Generator(Gs, batch_size=1, randomize_noise=False)
# 	return Gs

def generate_image(generator, latent_vector):
	latent_vector = latent_vector.reshape((1, 18, 512))
	generator.set_dlatents(latent_vector)
	img_array = generator.generate_images()[0]
	img = PIL.Image.fromarray(img_array, 'RGB')
	return img.resize((512, 512))   


# ENCODING

generate_inputs_1 = {
	'portrait': runway.image(),
	'iterations': runway.number(min=1, max=5000, default=10, step=1.0)
}
generate_outputs_1 = {
	"text": runway.text
}

encodeCount = 0
latent_vectors = []

@runway.command('encode', inputs=generate_inputs_1, outputs=generate_outputs_1)
def find_in_space(model, inputs):
	global generated_dlatents
	global prevIterations
	s2 = "Did not encode."
	if (inputs['iterations'] != prevIterations):
		generator.reset_dlatents()
		names = ["looking at you!"]
		perceptual_model.set_reference_images(inputs['portrait'])
		print ("image loaded.")
		prevIterations = inputs['iterations']
		print ("encoding for: ", prevIterations, " iterations.")
		op = perceptual_model.optimize(generator.dlatent_variable, iterations=inputs['iterations'], learning_rate=1.)
		# load latent vectors	
		generated_dlatents = generator.get_dlatents()
		pbar = tqdm(op, leave=False, total=inputs['iterations'], mininterval=2.0, miniters=10, disable=True)
		for loss in pbar:
			pbar.set_description(' '.join(names)+' Loss: %.2f' % loss)
		print(' '.join(names), ' loss:', loss)
		print (encodeCount, " finished encoding.") 
		s2 = "Finished encoding."
		latent_vectors.append(generator.get_dlatents())
		print(latent_vectors)
		encodeCount++
	return{"text": s2}


# GENERATION

cat = runway.category(choices=["1", "2", "3", "4"], default="1")

generate_inputs_2 = {
	'person': cat,
	'age': runway.number(min=-500, max=500, default=6, step=0.1)
}
generate_outputs_2 = {
	'image': runway.image(width=512, height=512)
}

@runway.command('generate', inputs=generate_inputs_2, outputs=generate_outputs_2)
def move_and_show(model, inputs):
	global latent_vector_2
	global latent_vector_1
	#latent_vector_1 = latent_vectors[int(inputs['person'])-1]	
	latent_vector = (latent_vector_1 + latent_vector_2) * 2
	# load direction
	age_direction = np.load('ffhq_dataset/latent_directions/age.npy')
	direction = age_direction
	# model = generator
	coeff = inputs['age']/5.0
	new_latent_vector = latent_vector.copy()
	new_latent_vector[:8] = (latent_vector + coeff*direction)[:8]
	image = (generate_image(model, new_latent_vector))
	#ax[i].set_title('Coeff: %0.1f' % coeff)
	#plt.show()
	return {'image': image}

if __name__ == '__main__':
	runway.run(debug=True)