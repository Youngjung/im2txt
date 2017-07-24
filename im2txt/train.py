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
import datetime
import numpy as np
from scipy.misc import imsave
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

	# Create training directory.
	train_dir = FLAGS.train_dir
	filename_saved_model = os.path.join(FLAGS.train_dir,'im2txt')
	if not tf.gfile.IsDirectory(train_dir):
		tf.logging.info("Creating training directory: %s", train_dir)
		tf.gfile.MakeDirs(train_dir)

	save_flags( os.path.join(FLAGS.train_dir,'flags.txt') )

	model_config = configuration.ModelConfig()
	model_config.input_file_pattern = FLAGS.input_file_pattern
	model_config.inception_checkpoint_file = FLAGS.inception_checkpoint_file
	training_config = configuration.TrainingConfig()

	vocab = vocabulary.Vocabulary( FLAGS.vocab_file )
	vocab_size = 12000

	# Build the TensorFlow graph.
	g = tf.Graph()
	with g.as_default():
		# Build the model (teacher-forcing mode).
		model = show_and_tell_model.ShowAndTellModel(
				model_config, mode="train", train_inception=FLAGS.train_inception)
		model.build()

		# Build the model (free-running mode).
		model_free = show_and_tell_model.ShowAndTellModel(
				model_config, mode="free", train_inception=FLAGS.train_inception, vocab=vocab, reuse=True )
		model_free.build([model.images,model.input_seqs,model.target_seqs,model.input_mask])

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
		
		# get free sentence
		free_softmax = model_free.inference_softmax
		free_softmax_reshaped = tf.reshape( free_softmax, [-1,30,vocab_size] )
		free_sentence = tf.argmax( free_softmax_reshaped, axis=2 )

		# prepare behavior to be LSTM's input
		teacher_behavior = tf.concat( [teacher_outputs,teacher_state_c,teacher_state_h], axis=1 )
		free_behavior = tf.concat( [free_outputs,free_state_c,free_state_h], axis=1 )

		d_lstm_cell = tf.contrib.rnn.BasicLSTMCell(model_config.num_lstm_units)
		d_lstm_cell = tf.contrib.rnn.DropoutWrapper(
							d_lstm_cell,
							input_keep_prob=model_config.lstm_dropout_keep_prob,
							output_keep_prob=model_config.lstm_dropout_keep_prob)

		with tf.variable_scope("discriminator") as scope_disc:
			teacher_lengths = tf.reduce_sum( model.input_mask, 1 )+2
			free_lengths = tf.ones_like(teacher_lengths)*(30+2)

			# run lstm
			d_outputs_teacher, _ = tf.nn.dynamic_rnn( cell=d_lstm_cell,
												inputs = teacher_behavior,
												sequence_length = teacher_lengths,
												dtype = tf.float32,
												scope = scope_disc )

			# gather last outputs (deals with variable length of captions)
			teacher_lengths = tf.expand_dims( teacher_lengths, 1 )
			batch_range = tf.expand_dims(tf.constant( np.array(range(model_config.batch_size)),dtype=tf.int32 ),1)
			gather_idx = tf.concat( [batch_range,teacher_lengths-1], axis=1 )
			d_last_output_teacher = tf.gather_nd( d_outputs_teacher, gather_idx )

			# FC to get T/F logits
			d_logits_teacher = tf.contrib.layers.fully_connected( inputs = d_last_output_teacher,
															num_outputs = 2,
															activation_fn = None,
															weights_initializer = model.initializer,
															scope = scope_disc )
			d_accuracy_teacher = tf.reduce_mean( tf.cast( tf.argmax( d_logits_teacher, axis=1 ), tf.float32 ) )

			scope_disc.reuse_variables()
			d_outputs_free, _ = tf.nn.dynamic_rnn( cell=d_lstm_cell,
												inputs = free_behavior,
												sequence_length = free_lengths,
												dtype = tf.float32,
												scope = scope_disc )
			d_last_output_free = d_outputs_free[:,-1,:]
			d_logits_free = tf.contrib.layers.fully_connected( inputs = d_last_output_free,
															num_outputs = 2,
															activation_fn = None,
															weights_initializer = model.initializer,
															scope = scope_disc )
			d_accuracy_free = tf.reduce_mean( tf.cast( 1-tf.argmax( d_logits_free, axis=1 ), tf.float32 ) )

			d_accuracy = ( d_accuracy_teacher + d_accuracy_free ) /2

		NLL_loss = model.total_loss

		d_loss_teacher = tf.reduce_mean( tf.nn.sigmoid_cross_entropy_with_logits(name='d_loss_teacher',
									 logits=d_logits_teacher, labels=tf.ones_like(d_logits_teacher) ) )
		d_loss_free = tf.reduce_mean( tf.nn.sigmoid_cross_entropy_with_logits(name='d_loss_free',
									 logits=d_logits_free, labels=tf.zeros_like(d_logits_free) ) )
		d_loss = d_loss_teacher + d_loss_free

		g_loss = tf.reduce_mean( tf.nn.sigmoid_cross_entropy_with_logits(name='g_loss',
									 logits=d_logits_free, labels=tf.ones_like(d_logits_free) ) )
		g_and_NLL_loss = g_loss + NLL_loss

		summary = {}
		summary['NLL_loss'] = tf.summary.scalar('NLL_loss', NLL_loss)
		summary['d_loss'] = tf.summary.scalar('d_loss', d_loss)
		summary['d_loss_teacher'] = tf.summary.scalar('d_loss_teacher', d_loss_teacher)
		summary['d_loss_free'] = tf.summary.scalar('d_loss_free', d_loss_free)
		summary['g_loss'] = tf.summary.scalar('g_loss', g_loss)
		summary['g_and_NLL_loss'] = tf.summary.scalar('g_and_NLL_loss', g_and_NLL_loss)
		summary['d_logits_free'] = tf.summary.histogram('d_logits_free', d_logits_free)
		summary['d_accuracy'] = tf.summary.histogram('d_accuracy', d_accuracy)

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
											loss = NLL_loss,
											global_step = model.global_step,
											learning_rate = learning_rate,
											optimizer = training_config.optimizer,
											clip_gradients = training_config.clip_gradients,
											learning_rate_decay_fn = learning_rate_decay_fn,
											variables = g_vars,
											name='optimize_NLL_loss' )

		train_op_disc = tf.contrib.layers.optimize_loss(
											loss = d_loss,
											global_step = model.global_step,
											learning_rate = learning_rate,
											optimizer = training_config.optimizer,
											clip_gradients = training_config.clip_gradients,
											learning_rate_decay_fn = learning_rate_decay_fn,
											variables = d_vars,
											name='optimize_disc_loss' )

		train_op_gen = tf.contrib.layers.optimize_loss(
											loss=NLL_loss+g_loss,
											global_step=model.global_step,
											learning_rate=learning_rate,
											optimizer=training_config.optimizer,
											clip_gradients=training_config.clip_gradients,
											learning_rate_decay_fn=learning_rate_decay_fn,
											variables = g_vars,
											name='optimize_gen_loss' )



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
			generator = caption_generator.CaptionGenerator( model_valid, vocab )

			try:
				# for validation
				f_valid_text = open(os.path.join(train_dir,'valid.txt'),'a')
				filenames = os.listdir('testimgs')
				valid_images = []
				for filename in filenames:
				    with tf.gfile.GFile(os.path.join('testimgs', filename),'r') as f:
				        valid_images.append( f.read() )
			
				# run inference for not-trained model
				#self.valid( valid_image, f_valid_text )
				for i, valid_image in enumerate(valid_images):
					captions = generator.beam_search( sess, valid_image )
					f_valid_text.write( 'initial caption (beam) {}\n'.format( str(datetime.datetime.now().time())[:-7] ) )
					for j, caption in enumerate(captions):
						sentence = [vocab.id_to_word(w) for w in caption.sentence[1:-1]]
						sentence = " ".join(sentence)
						sentence = "  {}-{}) {} (p={:.8f})".format(i,j, sentence, math.exp(caption.logprob))
						print( sentence )
						f_valid_text.write( sentence +'\n' )
				f_valid_text.flush()


				# run training loop
				lossnames_to_print = ['NLL_loss','g_loss', 'd_loss', 'd_acc', 'g_acc']
				val_NLL_loss = float('Inf')
				val_g_loss = float('Inf')
				val_d_loss = float('Inf')
				val_d_acc = 0
				val_g_acc = 0
				for epoch in range(FLAGS.number_of_steps):
					for batch_idx in range(nBatches):
						counter += 1
						is_disc_trained = False
						is_gen_trained = False

						# train NLL loss only (for im2txt sanity check)
						_, val_NLL_loss, smry_str, val_free_sentence, val_images, val_captions = sess.run(
							[train_op_NLL, NLL_loss, summary['NLL_loss'], free_sentence,model.images,model.input_seqs] )
						summaryWriter.add_summary(smry_str, counter)
						for i in range(32):
							filename='val_images_{}.jpg'.format(i)
							caption = [ vocab.id_to_word(v) for v in val_captions[i] ]
							caption = ' '.join(caption)
							print( '{}) {}'.format(i,caption) )
							imsave( filename, val_images[i] )
						pdb.set_trace()

#						if val_NLL_loss> 3.5:
#							_, val_NLL_loss, smry_str = sess.run([train_op_NLL, NLL_loss, summary['NLL_loss']] )
#							summaryWriter.add_summary(smry_str, counter)
#						else:
#							# train discriminator
#							_, val_d_loss, val_d_acc, \
#							smr1, smr2, smr3, smr4 = sess.run([train_op_disc, d_loss, d_accuracy, 
#								 summary['d_loss_teacher'], summary['d_loss_free'], summary['d_loss'],summary['d_accuracy']] )
#							summaryWriter.add_summary(smr1, counter)
#							summaryWriter.add_summary(smr2, counter)
#							summaryWriter.add_summary(smr3, counter)
#							summaryWriter.add_summary(smr4, counter)
#
#							# train generator
#							_, val_g_loss, val_NLL_loss, val_g_acc, smr1, smr2, smr3 = sess.run( 
#								[train_op_gen,g_loss,NLL_loss, d_accuracy, 
#								summary['g_loss'],summary['NLL_loss'], summary['g_and_NLL_loss']] )
#							summaryWriter.add_summary(smr1, counter)
#							summaryWriter.add_summary(smr2, counter)
#							summaryWriter.add_summary(smr3, counter)
#							_, val_g_loss, val_NLL_loss, val_g_acc, smr1, smr2, smr3 = sess.run( 
#								[train_op_gen,g_loss,NLL_loss, d_accuracy, 
#								summary['g_loss'],summary['NLL_loss'], summary['g_and_NLL_loss']] )
#							summaryWriter.add_summary(smr1, counter)
#							summaryWriter.add_summary(smr2, counter)
#							summaryWriter.add_summary(smr3, counter)
			
						if counter % FLAGS.log_every_n_steps==0:
							elapsed = time.time() - start_time
							log( epoch, batch_idx, nBatches, lossnames_to_print,
								 [val_NLL_loss,val_g_loss,val_d_loss,val_d_acc,val_g_acc], elapsed, counter )
			
						if counter % 500 == 1 or \
							(epoch==FLAGS.number_of_steps-1 and batch_idx==nBatches-1) :
							saver.save( sess, filename_saved_model, global_step=counter)
			
						if (batch_idx+1) % (nBatches//10) == 0  or batch_idx == nBatches-1:
							# run test after every epoch
							f_valid_text.write( 'count {} epoch {} batch {}/{} ({})\n'.format( \
									counter, epoch, batch_idx, nBatches, str(datetime.datetime.now().time())[:-7] ) )
							for i, valid_image in enumerate(valid_images):
								captions = generator.beam_search( sess, valid_image )
								for j, caption in enumerate(captions):
									sentence = [vocab.id_to_word(w) for w in caption.sentence[1:-1]]
									sentence = " ".join(sentence)
									sentence = "  {}-{}) {} (p={:.8f})".format(i,j, sentence, math.exp(caption.logprob))
									print( sentence )
									f_valid_text.write( sentence +'\n' )
								# free sentence check
							for i, caption in enumerate(val_free_sentence):
								sentence = [vocab.id_to_word(w) for w in caption[1:-1]]
								sentence = " ".join(sentence)
								sentence = "  free %d) %s" % (i, sentence)
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


