# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#		 http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Train the model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import time
import numpy as np
import tensorflow as tf

import configuration
import show_and_tell_model

FLAGS = tf.app.flags.FLAGS

tf.flags.DEFINE_string("input_file_pattern", "",
											 "File pattern of sharded TFRecord input files.")
tf.flags.DEFINE_string("inception_checkpoint_file", "",
											 "Path to a pretrained inception_v3 model.")
tf.flags.DEFINE_string("train_dir", "",
											 "Directory for saving and loading model checkpoints.")
tf.flags.DEFINE_boolean("train_inception", False,
												"Whether to train inception submodel variables.")
tf.flags.DEFINE_integer("number_of_steps", 1000000, "Number of training steps.")
tf.flags.DEFINE_integer("log_every_n_steps", 1,
												"Frequency at which loss and global step are logged.")

tf.logging.set_verbosity(tf.logging.INFO)


def main(unused_argv):
	assert FLAGS.input_file_pattern, "--input_file_pattern is required"
	assert FLAGS.train_dir, "--train_dir is required"

	model_config = configuration.ModelConfig()
	model_config.input_file_pattern = FLAGS.input_file_pattern
	model_config.inception_checkpoint_file = FLAGS.inception_checkpoint_file
	training_config = configuration.TrainingConfig()

	# Create training directory.
	train_dir = FLAGS.train_dir
	filename_saved_model = os.path.join(FLAGS.train_dir,'im2txt')
	if not tf.gfile.IsDirectory(train_dir):
		tf.logging.info("Creating training directory: %s", train_dir)
		tf.gfile.MakeDirs(train_dir)

	save_flags( os.path.join(FLAGS.train_dir,'flags.txt') )

	# Build the TensorFlow graph.
	g = tf.Graph()
	with g.as_default():
		# Build the model.
		model = show_and_tell_model.ShowAndTellModel(
				model_config, mode="train", train_inception=FLAGS.train_inception)
		model.build()

		# Set up the learning rate.
		learning_rate_decay_fn = None
		if FLAGS.train_inception:
			learning_rate = tf.constant(training_config.train_inception_learning_rate)
		else:
			learning_rate = tf.constant(training_config.initial_learning_rate)
			if training_config.learning_rate_decay_factor > 0:
				num_batches_per_epoch = (training_config.num_examples_per_epoch //
																 model_config.batch_size)
				decay_steps = int(num_batches_per_epoch *
													training_config.num_epochs_per_decay)

				def _learning_rate_decay_fn(learning_rate, global_step):
					return tf.train.exponential_decay(
							learning_rate,
							global_step,
							decay_steps=decay_steps,
							decay_rate=training_config.learning_rate_decay_factor,
							staircase=True)

				learning_rate_decay_fn = _learning_rate_decay_fn

		loss = { 'NLL' : model.total_loss }

		summary = { 'loss_NLL' : tf.summary.scalar('loss_NLL',loss['NLL']) }

		# Set up the training ops.
		train_op_NLL = tf.contrib.layers.optimize_loss(
				loss=loss['NLL'],
				global_step=model.global_step,
				learning_rate=learning_rate,
				optimizer=training_config.optimizer,
				clip_gradients=training_config.clip_gradients,
				learning_rate_decay_fn=learning_rate_decay_fn)

		# Set up the Saver for saving and restoring model checkpoints.
		saver = tf.train.Saver(max_to_keep=training_config.max_checkpoints_to_keep)

		with tf.Session() as sess:
			# load inception variables
			model.init_fn( sess )
			
			# Set up the training ops
			nBatches = num_batches_per_epoch
			
			summaryWriter = tf.summary.FileWriter(train_dir, sess.graph)
			tf.global_variables_initializer().run()
			
			# start input enqueue threads
			coord = tf.train.Coordinator()
			threads = tf.train.start_queue_runners(sess=sess, coord=coord)
			
			counter = 0
			start_time = time.time()
			could_load, checkpoint_counter = load( sess, saver, train_dir )
			if could_load:
				counter = checkpoint_counter
			try:
				# run training loop
				lossnames_to_print = ['NLL_loss']
				val_NLL_loss = float('Inf')
				for epoch in range(FLAGS.number_of_steps):
					for batch_idx in range(nBatches):
						counter += 1

						# train NLL loss only (for im2txt sanity check)
						_, val_NLL_loss, smry_str, = sess.run([train_op_NLL, loss['NLL'], summary['loss_NLL'] ] )
						summaryWriter.add_summary(smry_str,counter)

						if counter % FLAGS.log_every_n_steps==0:
							elapsed = time.time() - start_time
							log( epoch, batch_idx, nBatches, lossnames_to_print,[val_NLL_loss], elapsed, counter )
			
						if counter % 500 == 1 or (epoch==FLAGS.number_of_steps-1 and batch_idx==nBatches-1) :
							saver.save( sess, filename_saved_model, global_step=counter )
			except tf.errors.OutOfRangeError:
				print('Finished training: epoch limit reached')
			finally:
				coord.request_stop()
			coord.join(threads)

def load(sess, saver, checkpoint_dir):
	import re
	print(" [*] Reading checkpoints...")

	ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
	if ckpt and ckpt.model_checkpoint_path:
		ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
		saver.restore(sess, os.path.join(checkpoint_dir, ckpt_name))
		counter = int(next(re.finditer("(\d+)(?!.*\d)",ckpt_name)).group(0))
		print(" [*] Success to read {}".format(ckpt_name))
		return True, counter
	else:
		print(" [*] Failed to find a checkpoint")
		return False, 0

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
			str_losses += '{:.4f}'.format(loss)
		isFirst = False

	m,s = divmod( elapsed, 60 )
	h,m = divmod( m,60 )
	timestamp = "{:2}:{:02}:{:02}".format( int(h),int(m),int(s) )
	log = "{} e{} b {:>{}}/{} ({})=({})".format( timestamp, epoch, batch, nDigits, nBatches, str_lossnames, str_losses )
	if counter is not None:
		log = "{:>5}_".format(counter) + log
	print( log )
	if filelogger:
		filelogger.write( log )
	return log

def save_flags( path ):
    flags_dict = tf.flags.FLAGS.__flags
    with open(path, 'w') as f:
        for key,val in flags_dict.iteritems():
            f.write( '{} = {}\n'.format(key,val) )

if __name__ == "__main__":
	tf.app.run()
