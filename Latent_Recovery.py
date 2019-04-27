# Setup :)
import numpy as np
import time
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
tf.enable_eager_execution(config=config)

from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Flatten, Conv3D, Dense, Conv1D, Input, Reshape, Conv3DTranspose, BatchNormalization
from tensorflow.keras.losses import logcosh
from tensorflow.keras.regularizers import l2
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow import GradientTape
from tensorflow.train import Saver
from tensorflow import Session
from tensorflow.train import AdamOptimizer

from Voxel_Dataset import get_voxel_dataset
from Show_Voxel import *

import pdb

################################# AUTOENCODER ###############################
# (see Voxel_AE.py for more details)
voxel_input = Input(shape=(32, 32, 32, 1))
encode_c1 = Conv3D(8, kernel_size=3, activation='elu', padding='valid',
              data_format='channels_last', kernel_regularizer=l2(l=0.01))(voxel_input)
encode_b1 = BatchNormalization()(encode_c1)
encode_c2 = Conv3D(16, kernel_size=3, activation='elu', padding='same',
              strides=(2, 2, 2), kernel_regularizer=l2(l=0.01))(encode_b1)
encode_b2 = BatchNormalization()(encode_c2)
encode_c3 = Conv3D(32, kernel_size=3, activation='elu', padding='valid',
                  kernel_regularizer=l2(l=0.01))(encode_b2)
encode_b3 = BatchNormalization()(encode_c3)
encode_c4 = Conv3D(64, kernel_size=3, activation='elu', padding='same',
              strides=(2, 2, 2), kernel_regularizer=l2(l=0.01))(encode_b3)
encode_b4 = BatchNormalization()(encode_c4)
encode_f5 = Flatten()(encode_b4)
encode_b5 = BatchNormalization()(encode_f5)
latent = Dense(100, use_bias=True, activation='elu',
               kernel_regularizer=l2(l=0.01))(encode_b5)
encoder = Model(inputs=voxel_input, outputs=latent)
decoder_input = Input(shape=(100,))
decode_f1 = Dense(343, use_bias=True, activation='elu',
                 kernel_regularizer=l2(l=0.01))(decoder_input)
decode_b1 = BatchNormalization()(decode_f1)
decode_reshape = Reshape((7, 7, 7, 1), input_shape=(343,))(decode_b1)
decode_c2 = Conv3D(64, kernel_size=3, activation='elu',
                   padding='same', kernel_regularizer=l2(l=0.01))(decode_reshape)
decode_b2 = BatchNormalization()(decode_c2)
decode_c3 = Conv3DTranspose(32, kernel_size=3, activation='elu',padding='valid',
                   strides=(2, 2, 2), kernel_regularizer=l2(l=0.01))(decode_b2)
decode_b3 = BatchNormalization()(decode_c3)
decode_c4 = Conv3DTranspose(16, kernel_size=3, activation='elu',
                   padding='same', kernel_regularizer=l2(l=0.01))(decode_b3)
decode_b4 = BatchNormalization()(decode_c4)
decode_c5 = Conv3DTranspose(8, kernel_size=3, activation='elu', padding='valid',
                   strides=(2, 2, 2), output_padding=1, kernel_regularizer=l2(l=0.01))(decode_b4)
decode_b5 = BatchNormalization()(decode_c5)
decode_output = Conv3DTranspose(1, kernel_size=3, activation='sigmoid',
                   padding='same', kernel_regularizer=l2(l=0.01))(decode_b5)
decoder = Model(inputs=decoder_input, outputs=decode_output)
reconstruction = decoder(encoder(voxel_input))
ae = Model(inputs=voxel_input, outputs=reconstruction)
# ae.summary()

# Load already trained weights
model_checkpoint_file = 'model/ae_checkpoint'
ae.load_weights(model_checkpoint_file)

# Get Encoder portion of AE
encoder = ae.layers[1]
encoder.summary()

# Get Decoder portion of AE
decoder = ae.layers[2]
decoder.summary()

optimizer = AdamOptimizer(1e-4)

################### MODIFIED BINARY CROSS ENTROPY LOSS FUNCTION ############
# Binary cross entropy but with value clamping
def lambda_binary_crossentropy(y_true, y_pred):
  y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
  binary_entr = y_true*((-y_true * tf.log(y_pred)) - ((1.0-y_true) * tf.log(1.0-y_pred)))

  # getting tensor values into scalar
  loss = tf.reduce_sum(binary_entr, axis=1)
  scalar_loss = tf.reduce_mean(loss)

  return scalar_loss

# Gradient computation with given model, input latent space, and partial voxel
def compute_gradients(model, latent, partial):
  with GradientTape() as tape:
    tape.watch(latent)
    y = model(latent)

    partial = tf.cast(partial[0], dtype=tf.float32)
    loss = lambda_binary_crossentropy(partial, y)

    return tape.gradient(loss, latent), loss

#
# "Generates" a latent space by passing an initially random latent vector
# through the decoder netowrk, and computing loss between
# decoder's output and a partial view voxel.
#
# Runs updates until loss is under loss_thresh
#
def recover_latent(partial, loss_thresh, seed_latent_vector=None):
  # Latent vector starts as a random vector
  if seed_latent_vector is None:
    curr_latent = tf.random.uniform(shape=(1, 100))
    curr_latent = tf.Variable(curr_latent)
  else:
    curr_latent = tf.Variable(seed_latent_vector)

  current_prediction = decoder(curr_latent)
  current_prediction = np.reshape(current_prediction.numpy(), (32,32,32))
  plot_voxel(convert_to_sparse_voxel_grid(current_prediction), voxel_res=(32,32,32))

  loss = float('inf')
  iter = 0
  while loss > loss_thresh:
    print 'Iteration: ', str(iter)
    # Compute gradient and loss of current latent vector
    gradients, loss = compute_gradients(decoder, curr_latent, partial)
    #print loss
    gradients = tf.reshape(gradients, (1, 100))

    # Updating latent vector according to gradient
    optimizer.apply_gradients(zip([gradients], [curr_latent]))
    iter += 1

    if iter % 1000 == 0:
      print loss
      current_prediction = decoder(curr_latent)
      current_prediction = np.reshape(current_prediction.numpy(), (32,32,32))
      plot_voxel(convert_to_sparse_voxel_grid(current_prediction), voxel_res=(32,32,32))

# Testing on dummy voxel
#dummy_vox = tf.random.uniform(shape=(32, 32, 32))

# Testing dataset.
train_dataset, validate_dataset, test_dataset = get_voxel_dataset(batch_size=1, down_sample=True)

for train_x in train_dataset:
  train_x_partial = np.reshape(train_x[0][0][0].numpy(), (32,32,32))
  plot_voxel(convert_to_sparse_voxel_grid(train_x_partial), voxel_res=(32,32,32))

  use = raw_input("Use model? [y/n]")

  if use == "y":
    seed_latent_space = encoder(tf.reshape(tf.cast(train_x[0][0][0], dtype=tf.float32), shape=(1,32,32,32,1)))
    recover_latent(train_x, 1e-7, seed_latent_space)

  # current_prediction = ae(tf.cast(train_x[0], dtype=tf.float32))
  # current_prediction = np.reshape(current_prediction.numpy(), (32,32,32))
  # plot_voxel(convert_to_sparse_voxel_grid(current_prediction), voxel_res=(32,32,32))
  
