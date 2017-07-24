"""
Some codes from https://github.com/Newmu/dcgan_code
"""
from __future__ import division
import math
import json
import random
import pprint
import scipy.misc
import numpy as np
import copy
from time import gmtime, strftime
from six.moves import xrange
import pdb

import tensorflow as tf
import tensorflow.contrib.slim as slim
from utils import image_processing

pp = pprint.PrettyPrinter()

get_stddev = lambda x, k_h, k_w: 1/math.sqrt(k_w*k_h*x.get_shape()[-1])

def show_all_variables():
	model_vars = tf.trainable_variables()
	slim.model_analyzer.analyze_vars(model_vars, print_info=True)

def get_image(image_path, input_height, input_width,
							resize_height=64, resize_width=64,
							crop=True, grayscale=False):
	image = imread(image_path, grayscale)
	return transform(image, input_height, input_width,
									 resize_height, resize_width, crop)

def save_images(images, size, image_path):
	return imsave(inverse_transform(images), size, image_path)

def imread(path, grayscale = False):
	if (grayscale):
		return scipy.misc.imread(path, flatten = True).astype(np.float)
	else:
		return scipy.misc.imread(path, mode='RGB').astype(np.float)

def merge_images(images, size):
	return inverse_transform(images)

def merge(images, size):
	h, w = images.shape[1], images.shape[2]
	if (images.shape[3] in (3,4)):
		c = images.shape[3]
		img = np.zeros((h * size[0], w * size[1], c))
		for idx, image in enumerate(images):
			i = idx % size[1]
			j = idx // size[1]
			img[j * h:j * h + h, i * w:i * w + w, :] = image
		return img
	elif images.shape[3]==1:
		img = np.zeros((h * size[0], w * size[1]))
		for idx, image in enumerate(images):
			i = idx % size[1]
			j = idx // size[1]
			img[j * h:j * h + h, i * w:i * w + w] = image[:,:,0]
		return img
	else:
		raise ValueError('in merge(images,size) images parameter '
										 'must have dimensions: HxW or HxWx3 or HxWx4')

def imsave(images, size, path):
	image = np.squeeze(merge(images, size))
	return scipy.misc.imsave(path, image)

def center_crop(x, crop_h, crop_w,
								resize_h=64, resize_w=64):
	if crop_w is None:
		crop_w = crop_h
	h, w = x.shape[:2]
	j = int(round((h - crop_h)/2.))
	i = int(round((w - crop_w)/2.))
	return scipy.misc.imresize(
			x[j:j+crop_h, i:i+crop_w], [resize_h, resize_w])

def transform(image, input_height, input_width, 
							resize_height=64, resize_width=64, crop=True):
	if crop:
		cropped_image = center_crop(
			image, input_height, input_width, 
			resize_height, resize_width)
	else:
		cropped_image = scipy.misc.imresize(image, [resize_height, resize_width])
	return np.array(cropped_image)/127.5 - 1.

def inverse_transform(images):
	return (images+1.)/2.

def to_json(output_path, *layers):
	with open(output_path, "w") as layer_f:
		lines = ""
		for w, b, bn in layers:
			layer_idx = w.name.split('/')[0].split('h')[1]

			B = b.eval()

			if "lin/" in w.name:
				W = w.eval()
				depth = W.shape[1]
			else:
				W = np.rollaxis(w.eval(), 2, 0)
				depth = W.shape[0]

			biases = {"sy": 1, "sx": 1, "depth": depth, "w": ['%.2f' % elem for elem in list(B)]}
			if bn != None:
				gamma = bn.gamma.eval()
				beta = bn.beta.eval()

				gamma = {"sy": 1, "sx": 1, "depth": depth, "w": ['%.2f' % elem for elem in list(gamma)]}
				beta = {"sy": 1, "sx": 1, "depth": depth, "w": ['%.2f' % elem for elem in list(beta)]}
			else:
				gamma = {"sy": 1, "sx": 1, "depth": 0, "w": []}
				beta = {"sy": 1, "sx": 1, "depth": 0, "w": []}

			if "lin/" in w.name:
				fs = []
				for w in W.T:
					fs.append({"sy": 1, "sx": 1, "depth": W.shape[0], "w": ['%.2f' % elem for elem in list(w)]})

				lines += """
					var layer_%s = {
						"layer_type": "fc", 
						"sy": 1, "sx": 1, 
						"out_sx": 1, "out_sy": 1,
						"stride": 1, "pad": 0,
						"out_depth": %s, "in_depth": %s,
						"biases": %s,
						"gamma": %s,
						"beta": %s,
						"filters": %s
					};""" % (layer_idx.split('_')[0], W.shape[1], W.shape[0], biases, gamma, beta, fs)
			else:
				fs = []
				for w_ in W:
					fs.append({"sy": 5, "sx": 5, "depth": W.shape[3], "w": ['%.2f' % elem for elem in list(w_.flatten())]})

				lines += """
					var layer_%s = {
						"layer_type": "deconv", 
						"sy": 5, "sx": 5,
						"out_sx": %s, "out_sy": %s,
						"stride": 2, "pad": 1,
						"out_depth": %s, "in_depth": %s,
						"biases": %s,
						"gamma": %s,
						"beta": %s,
						"filters": %s
					};""" % (layer_idx, 2**(int(layer_idx)+2), 2**(int(layer_idx)+2),
							 W.shape[0], W.shape[3], biases, gamma, beta, fs)
		layer_f.write(" ".join(lines.replace("'","").split()))

def make_gif(images, fname, duration=2, true_image=False):
	import moviepy.editor as mpy

	def make_frame(t):
		try:
			x = images[int(len(images)/duration*t)]
		except:
			x = images[-1]

		if true_image:
			return x.astype(np.uint8)
		else:
			return ((x+1)/2*255).astype(np.uint8)

	clip = mpy.VideoClip(make_frame, duration=duration)
	clip.write_gif(fname, fps = len(images) / duration)

def visualize(sess, model, config, option):
	image_frame_dim = int(math.ceil(config.batch_size**.5))
	if option == 0:
		z_sample = np.random.uniform(-0.5, 0.5, size=(config.batch_size, model.z_dim))
		samples = sess.run(model.sampler, feed_dict={model.t_z: z_sample})
		save_images(samples, [image_frame_dim, image_frame_dim], './samples/test_%s.png' % strftime("%Y%m%d%H%M%S", gmtime()))
	elif option == 1:
		values = np.arange(0, 1, 1./config.batch_size)
		for idx in xrange(100):
			print(" [*] %d" % idx)
			z_sample = np.zeros([config.batch_size, model.z_dim])
			for kdx, z in enumerate(z_sample):
				z[idx] = values[kdx]

			if config.dataset == "mnist":
				y = np.random.choice(10, config.batch_size)
				y_one_hot = np.zeros((config.batch_size, 10))
				y_one_hot[np.arange(config.batch_size), y] = 1

				samples = sess.run(model.sampler, feed_dict={model.t_z: z_sample, model.y: y_one_hot})
			else:
				samples = sess.run(model.sampler, feed_dict={model.t_z: z_sample})

			save_images(samples, [image_frame_dim, image_frame_dim], './samples/test_arange_%s.png' % (idx))
	elif option == 2:
		values = np.arange(0, 1, 1./config.batch_size)
		for idx in [random.randint(0, 99) for _ in xrange(100)]:
			print(" [*] %d" % idx)
			z = np.random.uniform(-0.2, 0.2, size=(model.z_dim))
			z_sample = np.tile(z, (config.batch_size, 1))
			#z_sample = np.zeros([config.batch_size, model.z_dim])
			for kdx, z in enumerate(z_sample):
				z[idx] = values[kdx]

			if config.dataset == "mnist":
				y = np.random.choice(10, config.batch_size)
				y_one_hot = np.zeros((config.batch_size, 10))
				y_one_hot[np.arange(config.batch_size), y] = 1

				samples = sess.run(model.sampler, feed_dict={model.t_z: z_sample, model.y: y_one_hot})
			else:
				samples = sess.run(model.sampler, feed_dict={model.t_z: z_sample})

			try:
				make_gif(samples, './samples/test_gif_%s.gif' % (idx))
			except:
				save_images(samples, [image_frame_dim, image_frame_dim], './samples/test_%s.png' % strftime("%Y%m%d%H%M%S", gmtime()))
	elif option == 3:
		values = np.arange(0, 1, 1./config.batch_size)
		for idx in xrange(100):
			print(" [*] %d" % idx)
			z_sample = np.zeros([config.batch_size, model.z_dim])
			for kdx, z in enumerate(z_sample):
				z[idx] = values[kdx]

			samples = sess.run(model.sampler, feed_dict={model.t_z: z_sample})
			make_gif(samples, './samples/test_gif_%s.gif' % (idx))
	elif option == 4:
		image_set = []
		values = np.arange(0, 1, 1./config.batch_size)

		for idx in xrange(100):
			print(" [*] %d" % idx)
			z_sample = np.zeros([config.batch_size, model.z_dim])
			for kdx, z in enumerate(z_sample): z[idx] = values[kdx]

			image_set.append(sess.run(model.sampler, feed_dict={model.t_z: z_sample}))
			make_gif(image_set[-1], './samples/test_gif_%s.gif' % (idx))

		new_image_set = [merge(np.array([images[idx] for images in image_set]), [10, 10]) \
				for idx in range(64) + range(63, -1, -1)]
		make_gif(new_image_set, './samples/test_gif_merged.gif', duration=8)

# -----------------------------# new added functions for cyclegan
class ImagePool(object):
	def __init__(self, maxsize=50):
		self.maxsize = maxsize
		self.num_img = 0
		self.images = []
	def __call__(self, image):
		if self.maxsize == 0:
			return image
		if self.num_img < self.maxsize:
			self.images.append(image)
			self.num_img=self.num_img+1
			return image
		if np.random.rand() > 0.5:
			idx = int(np.random.rand()*self.maxsize)
			tmp = copy.copy(self.images[idx])
			self.images[idx] = image
			return tmp
		else:
			return image

def load_test_data(image_path, fine_size=256):
	img = imread(image_path)
	img = scipy.misc.imresize(img, [fine_size, fine_size])
	img = img/127.5 - 1
	return img

def gray2rgb_ifneeded( img ):
	if img.ndim == 2:
		img.resize(( img.shape[0], img.shape[1], 1) )
		return np.repeat( img, 3, 2 )
	else:
		return img
	
def load_data(image_path, flip=True, is_test=False):
	img_A, img_B = load_image(image_path)
	img_A = gray2rgb_ifneeded( img_A )
	img_B = gray2rgb_ifneeded( img_B )
	img_A, img_B = preprocess_A_and_B(img_A, img_B, flip=flip, is_test=is_test)

	img_A = img_A/127.5 - 1.
	img_B = img_B/127.5 - 1.

	try:
		img_AB = np.concatenate((img_A, img_B), axis=2)
	except:
		print( image_path )
		pdb.set_trace()
	# img_AB shape: (fine_size, fine_size, input_c_dim + output_c_dim)
	return img_AB

def load_image(image_path):
	img_A = imread(image_path[0])
	img_B = imread(image_path[1])
	return img_A, img_B

def preprocess_A_and_B(img_A, img_B, load_size=286, fine_size=256, flip=True, is_test=False):
	if is_test:
		img_A = scipy.misc.imresize(img_A, [fine_size, fine_size])
		img_B = scipy.misc.imresize(img_B, [fine_size, fine_size])
	else:
		img_A = scipy.misc.imresize(img_A, [load_size, load_size])
		img_B = scipy.misc.imresize(img_B, [load_size, load_size])

		h1 = int(np.ceil(np.random.uniform(1e-2, load_size-fine_size)))
		w1 = int(np.ceil(np.random.uniform(1e-2, load_size-fine_size)))
		img_A = img_A[h1:h1+fine_size, w1:w1+fine_size]
		img_B = img_B[h1:h1+fine_size, w1:w1+fine_size]

		if flip and np.random.random() > 0.5:
			img_A = np.fliplr(img_A)
			img_B = np.fliplr(img_B)

	return img_A, img_B

# -----------------------------

##########################
#	added by Youngjung
##########################

def log( epoch, batch, nBatches, lossnames, losses, elapsed, counter=None, filelogger=None ):
	nDigits = len(str(nBatches))
	str_lossnames = ""
	str_losses = ""
	assert( len(lossnames) == len(losses) )
	isFirst = True
	for lossname, loss in zip(lossnames,losses):
		if not isFirst:
			str_lossnames += ','
			str_losses += ', '
		str_lossnames += lossname
		if type(loss) == str:
			str_losses += loss
		else:
			str_losses += '{:.2f}'.format(loss)
		isFirst = False

	m,s = divmod( elapsed, 60 )
	h,m = divmod( m,60 )
	timestamp = "{:2}:{:02}:{:02}".format( int(h),int(m),int(s) )
	log = "{} epoch {} batch {:>{}}/{} ({})=({})".format( timestamp, epoch, batch, nDigits, nBatches, str_lossnames, str_losses )
	if counter is not None:
		log = "cnt{:>5} ".format(counter) + log
	print( log )
	if filelogger:
		filelogger.write( log )
	return log


def process_image(encoded_image, opts, thread_id=0):
	"""Decodes and processes an image string.

	Args:
		encoded_image: A scalar string Tensor; the encoded image.
		thread_id: Preprocessing thread id used to select the ordering of color
			distortions.

	Returns:
		A float32 Tensor of shape [height, width, 3]; the processed image.
	"""
	return image_processing.process_image(encoded_image,
											is_training=opts.phase=='train',
											height=opts.input_height,
											width=opts.input_width,
											thread_id=thread_id,
											image_format=opts.image_format)

def save_flags( path ):
	flags_dict = tf.flags.FLAGS.__flags
	with open(path, 'w') as f:
		for key,val in flags_dict:
			f.write( '{},{}\n'.format(key,val) )
