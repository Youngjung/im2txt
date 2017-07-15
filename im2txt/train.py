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

import pdb

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
	filename_saved_model = os.path.join(FLAGS.train_dir,'im2txt')
	if not tf.gfile.IsDirectory(train_dir):
		tf.logging.info("Creating training directory: %s", train_dir)
		tf.gfile.MakeDirs(train_dir)

	vocab = vocabulary.Vocabulary( FLAGS.vocab_file )

	# Build the TensorFlow graph.
	g = tf.Graph()
	with g.as_default():
		# Build the model (teacher-forcing mode).
		model = show_and_tell_model.ShowAndTellModel(
				model_config, mode="train", train_inception=FLAGS.train_inception)
		model.build()

		# Build the model (free-running mode).
		model_free = show_and_tell_model.ShowAndTellModel(
				model_config, mode="free", vocab=vocab, reuse=True )
		model_free.build()

		# Build the model for validation with variable sharing
		model_valid = show_and_tell_model.ShowAndTellModel(
				model_config, mode="inference", reuse=True )
		model_valid.build()

		# get teacher behavior
		teacher_outputs, [teacher_state_c,teacher_state_h] = model.behavior
		teacher_state_c = tf.expand_dims( teacher_state_c, axis=1 )
		teacher_state_h = tf.expand_dims( teacher_state_h, axis=1 )

		# get free behavior
		free_outputs, [free_state_c,free_state_h] = model_free.behavior
		free_state_c = tf.expand_dims( free_state_c, axis=1 )
		free_state_h = tf.expand_dims( free_state_h, axis=1 )
		
		# prepare behavior to be LSTM's input
		teacher_behavior = tf.concat( [teacher_outputs,teacher_state_c,teacher_state_h], axis=1 )
		free_behavior = tf.concat( [free_outputs,free_state_c,free_state_h], axis=1 )

		d_lstm_cell = tf.contrib.rnn.BasicLSTMCell(model_config.num_lstm_units)
		d_lstm_cell = tf.contrib.rnn.DropoutWrapper(
							d_lstm_cell,
							input_keep_prob=model_config.lstm_dropout_keep_prob,
							output_keep_prob=model_config.lstm_dropout_keep_prob)

		with tf.variable_scope("discriminator") as scope_disc:
			teacher_lengths = tf.reduce_sum( model.input_mask, 1 )
			d_outputs_teacher, _ = tf.nn.dynamic_rnn( cell=d_lstm_cell,
												inputs = teacher_behavior,
												sequence_length = teacher_lengths,
												dtype = tf.float32,
												scope = scope_disc )
			#d_outputs_teacher = tf.transpose( d_outputs_teacher, [1,0,2] )
			#d_last_output_teacher = tf.gather( d_outputs_teacher, int(d_outputs_teacher.get_shape()[0])-1 )
			d_last_output_teacher  = tf.slice( d_outputs_teacher, [0,-1,0],[-1,-1,-1] )
			d_logits_teacher = tf.contrib.layers.fully_connected( inputs = d_last_output_teacher,
															num_outputs = 2,
															activation_fn = None,
															weights_initializer = model.initializer,
															scope = scope_disc )

			scope_disc.reuse_variables()
			free_lengths = tf.ones_like(teacher_lengths)*30
			d_outputs_free, _ = tf.nn.dynamic_rnn( cell=d_lstm_cell,
												inputs = free_behavior,
												sequence_length = free_lengths,
												dtype = tf.float32,
												scope = scope_disc )
			#d_outputs_free = tf.transpose( d_outputs_free, [1,0,2] )
			#d_last_output_free = tf.gather( d_outputs_free, int(d_outputs_free.get_shape()[0])-1 )
			d_last_output_free = tf.slice( d_outputs_free, [0,-1,0],[-1,-1,-1] )
			d_logits_free = tf.contrib.layers.fully_connected( inputs = d_last_output_free,
															num_outputs = 2,
															activation_fn = None,
															weights_initializer = model.initializer,
															scope = scope_disc )

		d_loss_teacher = tf.reduce_mean( tf.nn.sigmoid_cross_entropy_with_logits(
									 logits=d_logits_teacher, labels=tf.ones_like(d_logits_teacher) ) )
		d_loss_free = tf.reduce_mean( tf.nn.sigmoid_cross_entropy_with_logits(
									 logits=d_logits_free, labels=tf.zeros_like(d_logits_free) ) )
		d_loss = d_loss_teacher + d_loss_free

		g_loss = tf.reduce_mean( tf.nn.sigmoid_cross_entropy_with_logits(
									 logits=d_logits_free, labels=tf.ones_like(d_logits_free) ) )

		summary = {}
		summary['NLL_loss'] = tf.summary.scalar('NLL_loss', model.total_loss)
		summary['d_loss'] = tf.summary.scalar('d_loss', d_loss)
		summary['d_loss_teacher'] = tf.summary.scalar('d_loss_teacher', d_loss_teacher)
		summary['d_loss_free'] = tf.summary.scalar('d_loss_free', d_loss_free)
		summary['g_loss'] = tf.summary.scalar('g_loss', g_loss)

		# Set up the learning rate for training ops
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

		# Collect trainable variables
		vars_all = [ v for v in tf.trainable_variables() if v not in model.inception_variables ]
		d_vars = [ v for v in vars_all if 'discr' in v.name ]
		g_vars = [ v for v in vars_all if 'discr' not in v.name ]

		# Set up the training ops.
		train_op_NLL = tf.contrib.layers.optimize_loss(
											loss = model.total_loss,
											global_step = model.global_step,
											learning_rate = learning_rate,
											optimizer = training_config.optimizer,
											clip_gradients = training_config.clip_gradients,
											learning_rate_decay_fn = learning_rate_decay_fn,
											variables = g_vars )

		train_op_disc = tf.contrib.layers.optimize_loss(
											loss = d_loss,
											global_step = model.global_step,
											learning_rate = learning_rate,
											optimizer = training_config.optimizer,
											clip_gradients = training_config.clip_gradients,
											learning_rate_decay_fn = learning_rate_decay_fn,
											variables = d_vars )

		train_op_gen = tf.contrib.layers.optimize_loss(
											loss=model.total_loss+g_loss,
											global_step=model.global_step,
											learning_rate=learning_rate,
											optimizer=training_config.optimizer,
											clip_gradients=training_config.clip_gradients,
											learning_rate_decay_fn=learning_rate_decay_fn,
											variables = g_vars)



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
				f_valid_text.write( 'initial caption\n' )
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
						_, g_loss, summary_str = sess.run([train_op_NLL, model.total_loss, summary['NLL_loss']] )
						summaryWriter.add_summary(summary_str, counter)
			
						if counter % FLAGS.log_every_n_steps==0:
							elapsed = time.time() - start_time
							log( epoch, batch_idx, nBatches, lossnames_to_print, [g_loss], elapsed, counter )
			
						if counter % 500 == 1 or \
							(epoch==FLAGS.number_of_steps-1 and batch_idx==nBatches-1) :
							saver.save( sess, filename_saved_model, global_step=counter)
			
						if (batch_idx+1) % (nBatches//10) == 0  or batch_idx == nBatches-1:
							# run test after every epoch
							#self.valid( valid_image, f_valid_text )
							captions = generator.beam_search( sess, image_valid )
							f_valid_text.write( 'epoch {} batch {}/{}\n'.format( epoch, batch_idx ) )
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


