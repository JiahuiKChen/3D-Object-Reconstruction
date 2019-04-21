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

# Get Decoder portion of AE
decoder = ae.layers[2]
decoder.summary()

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
#   input_tensor = tf.cast(x[0], dtype=tf.float32)
  with GradientTape() as tape:
    tape.watch(latent)
    y = model(latent)
    loss = lambda_binary_crossentropy(partial, y)

    return tape.gradient(loss, latent), loss

#
# "Generates" a latent space by passing an initially random latent vector
# through the decoder netowrk, and computing loss between
# decoder's output and a partial view voxel.
#
# Runs updates until loss is under loss_thresh
#
def recover_latent(partial, loss_thresh):
  # Latent vector starts as a random vector
  curr_latent = tf.random.uniform(shape=(1, 100))
  curr_latent = tf.Variable(curr_latent)

  loss = float('inf')
  iter = 0
  while loss > loss_thresh:
    print 'Iteration: ', str(iter)
    # Compute gradient and loss of current latent vector
    gradients, loss = compute_gradients(decoder, curr_latent, partial)
    print loss
    gradients = tf.reshape(gradients, (1, 100))

    # Updating latent vector according to gradient
    optimizer.apply_gradients(zip([gradients], [curr_latent]))
    iter += 1

# Testing on dummy voxel
dummy_vox = tf.random.uniform(shape=(32, 32, 32))

recover_latent(dummy_vox, 10.0)
