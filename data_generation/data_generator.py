import tensorflow as tf
import os
import pandas as pd
import functools
import random

LABEL_FILE = 'train.csv'
IMAGE_SHAPE = [224, 224]
NB_EPOCHS = None
NUM_PARALLEL_CALLS = 4
PREFETCH_BUFFER_SIZE = 2 ** 8
SHUFFLE_BUFFER_SIZE = 2 ** 8


def get_landmark_challange_generator(data_dir):
	"""
	Args:
		data_dir: path to temporary storage directory
		training: a Boolean; if true, we use the train set, otherwise the test set.
		nb_images: how many images and labels to generate
	:return: 
	"""
	def gen():
		files = os.listdir(data_dir)
		files = list(set(files) - set([LABEL_FILE]))
		label_path = data_dir + LABEL_FILE
		label_df = pd.read_csv(label_path)[['id', 'landmark_id']]
		label_df.set_index('id', drop=True, inplace=True)
		label_dict = label_df.to_dict()['landmark_id']
		image_names, labels = [], []
		for _file in files:
			label = label_dict.get(_file.split('.')[0], None)
			if label:
				image_names.append(_file)
				labels.append(label)
		# sampling
		data = list(zip(image_names, labels))
		random.shuffle(data)
		image_names, labels = list(zip(*data))
		image_paths = [data_dir + image_name for image_name in image_names]
		for image_path, label in zip(image_paths, labels):
			yield image_path, label
	return gen


def input_fn(batch_size, data_dir, is_training=True):
	# get the generator fn
	gen = get_landmark_challange_generator(data_dir)

	def _parse_function(filename, label):
		image_string = tf.read_file(filename)
		image_decoded = tf.image.decode_png(image_string)
		image_resized = tf.image.resize_images(image_decoded, IMAGE_SHAPE)
		return image_resized, label

	def _image_augmentation(image, is_training=True):
		"""Image augmentation: cropping, flipping, and color transforms. Normalization"""
		if is_training:
			image = tf.image.random_flip_left_right(image)
			image = tf.image.random_brightness(image, max_delta=32. / 255.)
			image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
			image = tf.image.random_hue(image, max_delta=0.2)
			image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
		image = tf.image.per_image_standardization(image)
		return image

	image_augmentation_fn = functools.partial(_image_augmentation, is_training=is_training)

	dataset = tf.data.Dataset.from_generator(generator=gen, output_types=(tf.string, tf.int32), output_shapes=(tf.TensorShape(None), tf.TensorShape(None)))
	dataset = dataset.map(_parse_function, num_parallel_calls=NUM_PARALLEL_CALLS)
	dataset = dataset.shuffle(buffer_size=SHUFFLE_BUFFER_SIZE)

	dataset = dataset.batch(batch_size=batch_size)
	dataset = dataset.prefetch(buffer_size=PREFETCH_BUFFER_SIZE)
	dataset = dataset.repeat(NB_EPOCHS)

	images, labels = dataset.make_one_shot_iterator().get_next()
	image_augmentation_fn = functools.partial(image_augmentation_fn, is_training=is_training)
	images = tf.map_fn(image_augmentation_fn, images)
	return images, labels


