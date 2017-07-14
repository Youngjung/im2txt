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
import tensorflow as tf

import configuration
import show_and_tell_model
from inference_utils import vocabulary, caption_generator

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
tf.flags.DEFINE_string("vocab_file", "", "")

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
	if not tf.gfile.IsDirectory(train_dir):
		tf.logging.info("Creating training directory: %s", train_dir)
		tf.gfile.MakeDirs(train_dir)

	# Build the TensorFlow graph.
	g = tf.Graph()
	with g.as_default():
		# Build the model.
		model = show_and_tell_model.ShowAndTellModel(
				model_config, mode="train", train_inception=FLAGS.train_inception)
		model.build()

		# Build the model for validation with variable sharing
		model_valid = show_and_tell_model.ShowAndTellModel(
				model_config, mode="inference", reuse=True )
		model_valid.build()
		vocab = vocabulary.Vocabulary( FLAGS.vocab_file )

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

		# Set up the training ops.
		train_op = tf.contrib.layers.optimize_loss(
				loss=model.total_loss,
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
			summary_loss = tf.summary.scalar('loss', model.total_loss)
			
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
			lossnames_to_print = ['g_loss']
			generator = caption_generator.CaptionGenerator( model_valid, vocab )

			try:
				# for validation
				with tf.gfile.GFile('data/mscoco/raw-data/val2014/COCO_val2014_000000224477.jpg','r') as f:
					image_valid = f.read()
				f_valid_text = open('valid.txt','w')
			
				# run inference for not-trained model
				#self.valid( valid_image, f_valid_text )
				captions = generator.beam_search( sess, image_valid )
				for i, caption in enumerate(captions):
					sentence = [vocab.id_to_word(w) for w in caption.sentence[1:-1]]
					sentence = " ".join(sentence)
					sentence = "  %d) %s (p=%f)" % (i, sentence, math.exp(caption.logprob))
					print( sentence )
					f_valid_text.write( sentence +'\n' )
				f_valid_text.flush()


				# run training loop
				for epoch in range(FLAGS.number_of_steps):
					for batch_idx in range(nBatches):
						counter += 1
						_, g_loss, summary_str = sess.run([train_op, model.total_loss, summary_loss] )
						summaryWriter.add_summary(summary_str, counter)
			
						if counter % FLAGS.log_every_n_steps==0:
							elapsed = time.time() - start_time
							log( epoch, batch_idx, nBatches, lossnames_to_print, [g_loss], elapsed, counter )
			
						if counter % 500 == 1 or \
							(epoch==FLAGS.number_of_steps-1 and batch_idx==nBatches-1) :
							saver.save( sess, train_dir, global_step=counter)
			
					# run test after every epoch
					#self.valid( valid_image, f_valid_text )
					captions = generator.beam_search( sess, image_valid )
					for i, caption in enumerate(captions):
						sentence = [vocab.id_to_word(w) for w in caption.sentence[1:-1]]
						sentence = " ".join(sentence)
						sentence = "  %d) %s (p=%f)" % (i, sentence, math.exp(caption.logprob))
						print( sentence )
						f_valid_text.write( sentence +'\n' )
					f_valid_text.flush()

			
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


if __name__ == "__main__":
	tf.app.run()


