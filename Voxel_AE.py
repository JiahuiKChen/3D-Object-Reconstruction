import numpy as np
import time
import tensorflow as tf
tf.enable_eager_execution()

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
from Show_Voxel import plot_voxel, convert_to_sparse_voxel_grid

# Debug:
import pdb

################### MODIFIED BINARY CROSS ENTROPY LOSS FUNCTION ############
# Binary cross entropy but with a lambda parameter that
# encourages false positives and discourage false negatives (because of the high
# amounts of blank voxels, without this term the model would output empty voxels)
def lambda_binary_crossentropy(y_true, y_pred):
  y_pred = tf.clip_by_value(y_pred, 0.1, 1.0 - 1e-7)
  binary_entr = (-0.98 * y_true * tf.log(y_pred)) - ((0.02) * (1.0-y_true) * tf.log(1.0-y_pred))

  # getting tensor values into scalar
  loss = tf.reduce_sum(binary_entr, axis=1)
  scalar_loss = tf.reduce_mean(loss)

  return scalar_loss

########################## ENCODER NETWORK #################################
# Input is 32x32x32 point cloud
voxel_input = Input(shape=(32, 32, 32, 1))

# First convolutional layer: outputs 30x30x30
encode_c1 = Conv3D(8, kernel_size=3, activation='elu', padding='valid',
              data_format='channels_last', kernel_regularizer=l2(l=0.01))(voxel_input)
encode_b1 = BatchNormalization()(encode_c1)

# Second convolutional layer: outputs 15x15x15 (downsamples via striding)
encode_c2 = Conv3D(16, kernel_size=3, activation='elu', padding='same',
              strides=(2, 2, 2), kernel_regularizer=l2(l=0.01))(encode_b1)
encode_b2 = BatchNormalization()(encode_c2)

# Third convolutional layer: outputs 13x13x13
encode_c3 = Conv3D(32, kernel_size=3, activation='elu', padding='valid',
                  kernel_regularizer=l2(l=0.01))(encode_b2)
encode_b3 = BatchNormalization()(encode_c3)

# Fourth convolutional layer: outputs 7x7x7 (downsamples via striding)
encode_c4 = Conv3D(64, kernel_size=3, activation='elu', padding='same',
              strides=(2, 2, 2), kernel_regularizer=l2(l=0.01))(encode_b3)
encode_b4 = BatchNormalization()(encode_c4)

# Fifth layer, fully connected: outputs 343
encode_f5 = Flatten()(encode_b4)
encode_b5 = BatchNormalization()(encode_f5)

# Sixth layer - LATENT LAYER: outputs 100
latent = Dense(100, use_bias=True, activation='elu',
               kernel_regularizer=l2(l=0.01))(encode_b5)

encoder = Model(inputs=voxel_input, outputs=latent)

# Run this to get summary of encoder's layers
encoder.summary()


########################## DECODER NETWORK #################################
# Latent space of 1D dimension 100 is input for decoder
decoder_input = Input(shape=(100,))

# First layer of decoder, fully connected layer: outputs 343
decode_f1 = Dense(343, use_bias=True, activation='elu',
                 kernel_regularizer=l2(l=0.01))(decoder_input)
decode_b1 = BatchNormalization()(decode_f1)

# Reshape layer from fully connected to 7x7x7
# must add spacial dimension for convolutions to work
decode_reshape = Reshape((7, 7, 7, 1), input_shape=(343,))(decode_b1)

# Second convolutional layer: convolutes fully connected layer into 7x7x7
decode_c2 = Conv3D(64, kernel_size=3, activation='elu',
                   padding='same', kernel_regularizer=l2(l=0.01))(decode_reshape)
decode_b2 = BatchNormalization()(decode_c2)

# Third layer (second convolutional layer): outputs 15x15x15
decode_c3 = Conv3DTranspose(32, kernel_size=3, activation='elu',padding='valid',
                   strides=(2, 2, 2), kernel_regularizer=l2(l=0.01))(decode_b2)
decode_b3 = BatchNormalization()(decode_c3)

# Fourth convolutional layer: outputs 15x15x15
decode_c4 = Conv3DTranspose(16, kernel_size=3, activation='elu',
                   padding='same', kernel_regularizer=l2(l=0.01))(decode_b3)
decode_b4 = BatchNormalization()(decode_c4)

# Fifth convolutional layer: outputs 32x32x32
decode_c5 = Conv3DTranspose(8, kernel_size=3, activation='elu', padding='valid',
                   strides=(2, 2, 2), output_padding=1, kernel_regularizer=l2(l=0.01))(decode_b4)
decode_b5 = BatchNormalization()(decode_c5)

# OUTPUT LAYERRRRRRRR!!! Sigmoid function to output probability each voxel is filled
decode_output = Conv3DTranspose(1, kernel_size=3, activation='sigmoid',
                   padding='same', kernel_regularizer=l2(l=0.01))(decode_b5)

decoder = Model(inputs=decoder_input, outputs=decode_output)

# Run this to get layer summary for decode_reshape
# decoder.summary()

################################# AUTOENCODER ###############################
reconstruction = decoder(encoder(voxel_input))
ae = Model(inputs=voxel_input, outputs=reconstruction)

ae.summary()

# Setup training using custom loss, and adam optimizer
# ae.compile(optimizer='adam', loss=lambda_binary_crossentropy)

TRAIN = True

if TRAIN:
  # iterator = voxel_dataset.make_one_shot_iterator()
  # next_element = iterator.get_next()

  # # Setup callbacks for saving and for viewing progress.
  # callbacks = [
  #   # Save model after every epoch.
  #   ModelCheckpoint("model/voxel_ae"),

  #   # Track progress w/ TensorBoard
  #   TensorBoard()
  # ]

  # # Train on voxel dataset!
  # ae.fit(voxel_dataset, epochs=10, callbacks=callbacks, batch_size=64, steps_per_epoch=steps_epoch, shuffle=False)

  # Settings:
  EPOCHS = 10

  # Use Adam optimizer.
  optimizer = AdamOptimizer(1e-4)

  # Gradient computation.
  def compute_gradients(model, x):
    input_tensor = tf.cast(x[0], dtype=tf.float32)
    with GradientTape() as tape:
      y = model(input_tensor)
      loss = lambda_binary_crossentropy(input_tensor, y)
      return tape.gradient(loss, model.trainable_variables), loss

  # Apply gradient update to our model.
  def apply_gradients(optimizer, gradients, variables):
    optimizer.apply_gradients(zip(gradients, variables))

  # Setup logging.
  summary_writer = tf.contrib.summary.create_file_writer('./logs')

  i = 0
  for epoch in range(1, EPOCHS + 1):
    # Run through dataset doing batch updates.
    print "Epoch: ", epoch
    t0 = time.time()

    # Gets newly shuffled dataset of voxels each epochs
    voxel_dataset, steps_epoch = get_voxel_dataset(batch_size=128)

    # Trains over all batches of the shuffled dataset
    for train_x in voxel_dataset:
      gradients, loss = compute_gradients(ae, train_x)
      #print i, ",", loss
      apply_gradients(optimizer, gradients, ae.trainable_variables)

      # Track loss through time.
      if i % 100 == 0:
        with summary_writer.as_default():
          with tf.contrib.summary.always_record_summaries():
            tf.contrib.summary.scalar("loss", loss, step=i)
      i += 1


    t1 = time.time()
    epoch_time = t1 - t0

    # Track time for epochs.
    with summary_writer.as_default():
      with tf.contrib.summary.always_record_summaries():
        tf.contrib.summary.scalar("epoch_time", epoch_time, step = epoch)
        tf.contrib.summary.scalar("epoch_loss", loss, step=epoch)

    # Checkpoint model.
    ae.save_weights('model/ae_checkpoint')

    # TODO: Validation set?
else:
  # Load model from checkpoint.
  ae.load_weights('model/ae_checkpoint')

  # Get tf.data.Dataset of our voxels.
  voxel_dataset, steps_epoch = get_voxel_dataset(batch_size=1)

  for train_x in voxel_dataset:
    # View input.
    input_tensor = tf.cast(train_x[0], dtype=tf.float32)
    input_npy = input_tensor.numpy().reshape((32,32,32))
    input_npy = (input_npy + 1) // 3
    input_sparse = convert_to_sparse_voxel_grid(input_npy)
    plot_voxel(input_sparse, voxel_res=(32,32,32))

    # Generate and view reconstruction.
    y = ae(input_tensor)
    y = y.numpy().reshape((32,32,32))
    y_sparse = convert_to_sparse_voxel_grid(y)
    plot_voxel(y_sparse, voxel_res=(32,32,32))
