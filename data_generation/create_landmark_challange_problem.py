
import os
import pandas as pd
import matplotlib.pylab as plt
import random
from tensor2tensor.data_generators import image_utils
from tensor2tensor.utils import registry

LABEL_FILE = 'train.csv'


def landmark_challange_generator(data_dir, nb_images):
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
		images, labels = list(zip(*data))
		images, labels = images[:nb_images], labels[:nb_images]
		print(len(images))
		images = [plt.imread(data_dir + image_name) for image_name in image_names]
		for image, label in zip(images, labels):
			yield image_utils.image_generator(image, label)

	return gen


@registry.register_problem
class LandmarkChallenge(image_utils.Image2ClassProblem):
	"""
	Landmark challange
	https://www.kaggle.com/c/landmark-recognition-challenge
	"""

	@property
	def is_small(self):
		return True

	@property
	def num_classes(self):
		return 14951

	@property
	def class_labels(self):
		return [str(c) for c in range(self.num_classes)]

	@property
	def train_shards(self):
		return 10

	def generator(self, data_dir, tmp_dir, is_training=True):
		if is_training:
			return landmark_challange_generator(data_dir, 60000)
		else:
			return landmark_challange_generator(data_dir, 10000)
