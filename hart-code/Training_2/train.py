from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from init import *

import tensorflow as tf
import cnn
import reader

frames_per_epoch = reader.get_total_images()
steps_per_epoch = int(float(frames_per_epoch) / BATCH_SIZE)

print("Frames Per Epoch:", frames_per_epoch, "Steps Per Epoch", steps_per_epoch)

def train():
	user_input = ''
	while (user_input != 'n' and user_input != 'y'):
		user_input = input('Restore from checkpoint? y/n: ')

	do_load_model = user_input == 'y'
	"""Train CIFAR-10 for a number of steps."""
	with tf.Graph().as_default():

		current_epoch = tf.Variable(0)
		is_training = tf.placeholder(tf.bool)
		global_step = tf.contrib.framework.get_or_create_global_step()

		# Force input pipeline to CPU:0 to avoid operations sometimes ending up on
		# GPU and resulting in a slow down.

		with tf.device('/cpu:0'):
			images, labels = reader.inputs('train', batch_size=BATCH_SIZE, num_epochs=NUM_EPOCHS)

		with tf.Session() as sess:
			# Initialize the variables (the trained variables and the epoch counter).
			init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
			sess.run(init_op)

			model = cnn.init_model(sess, do_load_model)

			coord = tf.train.Coordinator()
			threads = tf.train.start_queue_runners(sess=sess, coord=coord)

			global_loss = 100


			try:
				losses = 0
				step = 0
				total_duration = 0
				print("still working...")
				while not coord.should_stop():
					img, lbl = sess.run([images, labels])
					if AUGMENT:
						img, lbl = reader.augment(img,lbl)
					# else:
					# 	img = img.astype(np.float32)
					# 	img = img / 255.0

					start_time = time.time()
					loss_value, accuracy = model.train_on_batch(img, lbl)

					losses += loss_value
					duration = time.time() - start_time
					total_duration += duration
					current_epoch = int(step / steps_per_epoch)
					lr = model.optimizer.lr
					decay = model.optimizer.decay
					iterations = model.optimizer.iterations
					lr_with_decay = lr / (1. + decay * K.cast(iterations, K.dtype(decay)))
					loss_str = " Learning Rate: " + str(K.eval(lr_with_decay))
					print(loss_str)

					print('Epoch: %d Step %d: loss = %.8f (%.3f sec)' % (current_epoch, step, loss_value, duration), end="\r")

					if step % 100 == 0:
						#print (model.predict(img)[0])
						if step > 0:
							losses = losses / 100.0

						if losses < global_loss:
							print('Epoch: %d Step %d: loss = %.8f (%.3f sec)' % (current_epoch, step, losses, total_duration))
							global_loss = losses
							cnn.save_model(model)

						losses = 0
						total_duration = 0

					step += 1

					if current_epoch > NUM_EPOCHS:
						coord.request_stop()

			except tf.errors.OutOfRangeError:
				print('***Out of Range Error***')

			finally:
				# When done, ask the threads to stop.
				coord.request_stop()
			print('Done training for %d epochs, %d steps.' % (NUM_EPOCHS, step))
		#End Session
	# End Graph


def main(argv=None):  # pylint: disable=unused-argument
	train()


if __name__ == '__main__':
	tf.app.run()
