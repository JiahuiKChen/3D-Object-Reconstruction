import numpy as np
import time
import os
import tensorflow as tf
tf.enable_eager_execution()

from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Flatten, Conv3D, Dense, Conv1D, Input, Reshape, Conv3DTranspose, BatchNormalization
from tensorflow.keras.losses import mean_squared_error
from tensorflow.keras.regularizers import l2
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow import GradientTape
from tensorflow.train import Saver
from tensorflow import Session
from tensorflow.train import AdamOptimizer

from Voxel_Dataset import get_voxel_dataset
from Show_Voxel import plot_voxel, convert_to_sparse_voxel_grid
from Helper import get_f1, get_loss, get_latent_loss
from Model_AE import get_voxel_ae

import pdb

# Gradient computation.
def compute_gradients(encoder, x, full_latent):
  with GradientTape() as tape:
    y = encoder(x)
    loss = mean_squared_error(full_latent, y)
    return tape.gradient(loss, encoder.trainable_variables), loss

# Apply gradient update to our model.
def apply_gradients(optimizer, gradients, variables):
  optimizer.apply_gradients(zip(gradients, variables))

# Trains just the encoder of the VAE, given a training set of partial view voxels
# "Labels" are the latent space of full view voxels
def train_partial_to_full(model_name, epochs, batch_size, load_weights_file,  verbose):
    model_checkpoint_file = 'model/' + model_name + '_ae_checkpoint'
    model_logs_folder = 'logs/' + model_name

    if not os.path.exists(model_logs_folder):
        os.makedirs(model_logs_folder)

    # Create the VAE, with saved weights from training on the full voxel
    ae = get_voxel_ae(load_weights_file, verbose)
    # Gets just the encoder, this is used to generate labels/latents from fulls
    full_encoder = ae.layers[1]

    # Gets encoder that we will be updating to go from partial to full_latent
    ae_1 = get_voxel_ae(load_weights_file, verbose)
    partial_encoder = ae_1.layers[1]

    # Use Adam optimizer.
    optimizer = AdamOptimizer(1e-4)

    # Setup logging.
    summary_writer = tf.contrib.summary.create_file_writer(model_logs_folder)

    def update_tensorboard(train_dataset, validation_dataset):
      # Calculate losses/f1 at every epoch.
      train_loss = get_latent_loss(train_dataset, partial_encoder, full_encoder, 100)
      validate_loss = get_latent_loss(validation_dataset, partial_encoder, full_encoder)

      # Write to Tensorboard.
      with summary_writer.as_default():
        with tf.contrib.summary.always_record_summaries():
          tf.contrib.summary.scalar("train_loss", train_loss, step=epoch)
          tf.contrib.summary.scalar("validate_loss", validate_loss, step=epoch)

    for epoch in range(1, epochs + 1):
      # Run through dataset doing batch updates.
      print "Epoch: ", epoch

      # Gets newly shuf
      train_dataset, validation_dataset, test_dataset = get_voxel_dataset(batch_size, down_sample=True)
      update_tensorboard(train_dataset, validation_dataset)

      # Trains over all batches of the shuffled dataset
      for train_x in train_dataset:
        partial_tensor = tf.cast(train_x[0][:,0,:,:,:,:], dtype=tf.float32)
        full_tensor = tf.cast(train_x[0][:,1,:,:,:,:], dtype=tf.float32)
        full_latent = full_encoder(full_tensor)
        gradients, loss = compute_gradients(partial_encoder, partial_tensor, full_latent)
        apply_gradients(optimizer, gradients, partial_encoder.trainable_variables)

      # Checkpoint model.
      partial_encoder.save_weights(model_checkpoint_file)
