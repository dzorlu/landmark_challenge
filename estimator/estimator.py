import tensorflow as tf
import tensorflow_hub as hub
import argparse
import sys
import os


sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from data_generation.data_generator import input_fn


FLAGS = tf.flags.FLAGS
parser = argparse.ArgumentParser()
parser.add_argument('--predict', action='store_true', default=False, help='predict')
parser.add_argument('--batch_size', default=32, type=int, help='batch size')
parser.add_argument('--number_classes', default=2, type=int, help='number of classes')
parser.add_argument('--model_dir', default='/tmp/tf/seg/', type=str, help='model directory')
parser.add_argument('--data_dir', default=None, type=str, help='data directory')
parser.add_argument('--train_steps', default=10000, type=int, help='number of training steps')

# Learning hyperaparmeters
_BASE_LR = 0.1
_LR_SCHEDULE = [  # (LR multiplier, epoch to start)
	(1.0 / 6, 0), (2.0 / 6, 1), (3.0 / 6, 2), (4.0 / 6, 3), (5.0 / 6, 4),
	(1.0, 5), (0.1, 30), (0.01, 60), (0.001, 80), (0.0001, 90)
]
THRESHOLD = 0.10

TRAINED_MODEL_PATH = "https://tfhub.dev/google/imagenet/mobilenet_v2_140_224/feature_vector/1"
TRAIN_IMAGE_SIZE = 224


def add_summaries(predictions, features, loss):
	"""
	Add summaries for images, variables and losses.
	"""
	global_summaries = set([])
	for model_var in tf.get_collection('trainable_variables'):
		global_summaries.add(tf.summary.histogram(model_var.op.name, model_var))
	# total loss
	global_summaries.add(tf.summary.scalar('loss', loss))
	summary_op = tf.summary.merge(list(global_summaries), name='summary_op')
	return summary_op


def learning_rate_schedule(current_epoch):
	"""Handles linear scaling rule, gradual warmup, and LR decay."""
	scaled_lr = _BASE_LR * (FLAGS.train_batch_size / 256.0)

	decay_rate = scaled_lr
	for mult, start_epoch in _LR_SCHEDULE:
		decay_rate = tf.where(current_epoch < start_epoch, decay_rate,
		                      scaled_lr * mult)

	return decay_rate


def model_fn(features, labels, mode, params):
	"""Model function for DenseNet classifier.

	Args:
	  features: inputs.
	  labels: one hot encoded classes
	  mode: one of tf.estimator.ModeKeys.{TRAIN, INFER, EVAL}
	  params: a parameter dictionary with the following keys: 

	Returns:
	  ModelFnOps for Estimator API.
	"""
	number_classes = params.get('number_classes')
	tf.logging.info("features tensor {}".format(features))
	tf.logging.info("labels tensor {}".format(labels))

	module = hub.Module(TRAINED_MODEL_PATH, trainable=(mode == tf.estimator.ModeKeys.TRAIN))
	features = module(features)
	logits = tf.layers.dense(features, units=number_classes, trainable=(mode == tf.estimator.ModeKeys.TRAIN))
	probs = tf.nn.softmax(logits)
	predicted_classes = tf.argmax(probs, axis=1)
	if mode == tf.estimator.ModeKeys.PREDICT:
		tf.logging.info("Starting to predict..")
		predictions = {
			'probabilities': probs,
		}
		return tf.estimator.EstimatorSpec(mode, predictions=predictions)

	# Add the loss.
	# Calculate loss, which includes softmax cross entropy and L2 regularization.
	# wraps the softmax_with_entropy fn. adds it to loss collection
	one_hot_labels = tf.one_hot(labels, depth=number_classes)
	tf.losses.softmax_cross_entropy(
		logits=logits, onehot_labels=one_hot_labels)
	loss = tf.losses.get_total_loss()

	if mode == tf.estimator.ModeKeys.EVAL:
		tf.logging.info("Starting to eval..")
		# Compute evaluation metrics.
		accuracy = tf.metrics.accuracy(labels=labels, predictions=predicted_classes, name='acc_op')
		metrics = {'accuracy': accuracy}
		tf.summary.scalar('accuracy', accuracy[1])
		return tf.estimator.EstimatorSpec(
			mode, loss=loss, eval_metric_ops=metrics)

	assert mode == tf.estimator.ModeKeys.TRAIN
	tf.logging.info("Starting to train..")
	global_step = tf.train.get_global_step()
	update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
	optimizer = tf.train.AdagradOptimizer(learning_rate=1e-4)
	with tf.control_dependencies(update_ops):
		train_op = optimizer.minimize(loss, global_step=global_step)
	tf.logging.info("predicted_classess {}".format(predicted_classes))
	add_summaries(predicted_classes, features, loss)
	return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)


def main(argv):
	args = parser.parse_args(argv[1:])
	print(args)
	predict = args.predict
	config = tf.estimator.RunConfig(
		save_checkpoints_steps=100,
		save_summary_steps=50,
		keep_checkpoint_max=2,
		model_dir=args.model_dir
	)
	classifier = tf.estimator.Estimator(
		model_fn=model_fn,
		config=config,
		params={
			'number_classes': args.number_classes,
		}
	)

	# # # Train the Model.
	if not predict:
		classifier.train(
			input_fn=lambda: input_fn(args.batch_size, args.data_dir),
			steps=args.train_steps)
		# # Eval the Model.
		# classifier.evaluate(
		# 	input_fn=lambda: input_fn(args.batch_size, args.data_dir))
	# else:
		# Predict the model.
		# pred_result = classifier.predict(
		# 	input_fn=lambda: generate_predict_data())

if __name__ == '__main__':
	tf.logging.set_verbosity(tf.logging.INFO)
	tf.app.run(main)

