# https://colab.research.google.com/drive/1RUaVUqCvyojwoMglp6cFoLDnCfLHBZtB#scrollTo=PVYrjvgE_8AU

import os
import pickle
import PIL.Image
import numpy as np
import dnnlib
import dnnlib.tflib as tflib
import config
from encoder.generator_model import Generator
import matplotlib.pyplot as plt
%matplotlib inline
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
    global Gs
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
        ax[i].imshow(generate_image(new_latent_vector))
        ax[i].set_title('Coeff: %0.1f' % coeff)
    [x.axis('off') for x in ax]
    #plt.show()
    output = fig2data(plt)
    return {'image': output}


def generate_image(latent_vector):
    latent_vector = latent_vector.reshape((1, 18, 512))
    generator.set_dlatents(latent_vector)
    img_array = generator.generate_images()[0]
    img = PIL.Image.fromarray(img_array, 'RGB')
    return img.resize((512, 512))    


def fig2data ( fig ):
    """
    @brief Convert a Matplotlib figure to a 4D numpy array with RGBA channels and return it
    @param fig a matplotlib figure
    @return a numpy 3D array of RGBA values
    """
    # draw the renderer
    fig.canvas.draw ( )
 
    # Get the RGBA buffer from the figure
    w,h = fig.canvas.get_width_height()
    buf = numpy.fromstring ( fig.canvas.tostring_argb(), dtype=numpy.uint8 )
    buf.shape = ( w, h,4 )
 
    # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
    buf = numpy.roll ( buf, 3, axis = 2 )
    return buf

if __name__ == '__main__':
    runway.run()