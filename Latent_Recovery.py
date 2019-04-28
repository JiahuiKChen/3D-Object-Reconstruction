import numpy as np
import time
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
tf.enable_eager_execution(config=config)

from tensorflow.train import AdamOptimizer

from Voxel_Dataset import get_voxel_dataset
from Show_Voxel import plot_voxel, convert_to_sparse_voxel_grid
from Helper import get_f1, get_loss, lambda_binary_crossentropy, get_latent_loss
from Model_AE import get_voxel_ae

import pdb

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
def latent_optimization(partial, decoder, iterations=1000, seed_latent_vector=None, verbose=False):
  # Latent vector starts as a random vector
  if seed_latent_vector is None:
    curr_latent = tf.random.uniform(shape=(1, 100))
    curr_latent = tf.Variable(curr_latent)
  else:
    curr_latent = tf.Variable(seed_latent_vector)

  current_prediction = decoder(curr_latent)

  for i in range(iterations):
    # Compute gradient and loss of current latent vector
    gradients, loss = compute_gradients(decoder, curr_latent, partial)
    gradients = tf.reshape(gradients, (1, 100))

    # Updating latent vector according to gradient
    optimizer.apply_gradients(zip([gradients], [curr_latent]))

  return curr_latent

def latent_recovery(full_load_weights_file, partial_load_weights_file=None, optimize_latent=False, verbose=False):
  
  # Get model loaded with weights from training on full views 
  ae = get_voxel_ae(full_load_weights_file, verbose)

  # Full encoder for latent recovery metrics.
  ae_1 = get_voxel_ae(full_load_weights_file, verbose)
  full_encoder = ae.layers[1]
  
  # Get Encoder portion of AE, if partial weights exist then load them into encoder 
  encoder = ae.layers[1]
  if partial_load_weights_file != None:
    encoder.load_weights(partial_model_checkpoint_file)
  
  # Get Decoder portion of AE
  decoder = ae.layers[2]

  optimizer = AdamOptimizer(1e-4)

  def encode_latent(x):
    '''
    Given input partial view tensor, derive latent space.
    '''

    latent_space = encoder(x)
    if optimize_latent:
      latent_space = recover_latent(x, decoder, seed_latent_vector=latent_space)

    return latent_space

  def reconstruct(x):
    '''
    Given input partial view tensor, derive reconstructed voxel tensor.
    '''
    return decoder(encode_latent(x))
  
  # Testing dataset.
  train_dataset, validate_dataset, test_dataset = get_voxel_dataset(batch_size=1, down_sample=True)
  
  # Go through datasets and calculate metrics.
  mse_train = get_latent_loss(train_dataset, encode_latent, full_encoder)
  mse_validate = get_latent_loss(validate_dataset, encode_latent, full_encoder)
  mse_test = get_latent_loss(test_dataset, encode_latent, full_encoder)
  f1_train = get_f1(train_dataset, reconstruct, 0.5, partial=True)
  f1_validate = get_f1(validate_dataset, reconstruct, 0.5, partial=True)
  f1_test = get_f1(test_dataset, reconstruct, 0.5, partial=True)

  print "MSE: Train: ", mse_train, ", Validate: ", mse_validate, ", Test: ", mse_test, "."
  print "F1: Train: ", f1_train, ", Validate: ", f1_validate, ", Test: ", f1_test, "."  
  
  # TODO: YCB
