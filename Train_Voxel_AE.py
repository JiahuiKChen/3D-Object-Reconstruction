import numpy as np
import time
import os
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
from Helper import get_f1, get_loss, lambda_binary_crossentropy
from Model_AE import get_voxel_ae

# Debug:
import pdb
  
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

def train_voxel_ae(model_name, epochs, batch_size, load_weights_file,  verbose):
  '''
  Train a voxel as specified.
  '''
  
  model_checkpoint_file = 'model/' + model_name + '_ae_checkpoint'
  model_logs_folder = 'logs/' + model_name

  if not os.path.exists(model_logs_folder):
    os.makedirs(model_logs_folder)

  ae = get_voxel_ae(load_weights_file, verbose)
    
  # Use Adam optimizer.
  optimizer = AdamOptimizer(1e-4)

  # Setup logging.
  summary_writer = tf.contrib.summary.create_file_writer(model_logs_folder)

  def update_tensorboard(train_dataset, validation_dataset):
    # Calculate losses/f1 at every epoch.
    train_loss = get_loss(train_dataset, ae, 100)
    train_f1 = get_f1(train_dataset, ae, 0.5, 100)
    validate_loss = get_loss(validation_dataset, ae)
    validate_f1 = get_f1(validation_dataset, ae, 0.5)

    # Write to Tensorboard.
    with summary_writer.as_default():
      with tf.contrib.summary.always_record_summaries():
        tf.contrib.summary.scalar("train_loss", train_loss, step=epoch)
        tf.contrib.summary.scalar("train_f1", train_f1, step=epoch)
        tf.contrib.summary.scalar("validate_loss", validate_loss, step=epoch)
        tf.contrib.summary.scalar("validate_f1", validate_f1, step=epoch)
        
  for epoch in range(1, epochs + 1):
    # Run through dataset doing batch updates.
    print "Epoch: ", epoch

    # Gets newly shuf
    train_dataset, validation_dataset, test_dataset = get_voxel_dataset(batch_size)
    update_tensorboard(train_dataset, validation_dataset)

    # Trains over all batches of the shuffled dataset
    for train_x in train_dataset:
      gradients, loss = compute_gradients(ae, train_x)
      apply_gradients(optimizer, gradients, ae.trainable_variables)
      
    # Checkpoint model.
    ae.save_weights(model_checkpoint_file)
