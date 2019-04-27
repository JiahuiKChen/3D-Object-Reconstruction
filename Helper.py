import numpy as np
import time
import os
import copy
import tensorflow as tf
tf.enable_eager_execution()

from tensorflow.keras.losses import mean_squared_error

import pdb

################### MODIFIED BINARY CROSS ENTROPY LOSS FUNCTION ############
def lambda_binary_crossentropy(y_true, y_pred):
  '''
  Binary cross entropy but with a lambda parameter that
  encourages false positives and discourage false negatives (because of the high
  amounts of blank voxels, without this term the model would output empty voxels)
  '''
  y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
  binary_entr = (-y_true * tf.log(y_pred)) - ((1.0-y_true) * tf.log(1.0-y_pred))

  # getting tensor values into scalar
  loss = tf.reduce_sum(binary_entr, axis=1)
  scalar_loss = tf.reduce_mean(loss)

  return scalar_loss

def f1_prediction(y_true, y_pred, threshold):
  '''
  TP,TN,FP,FN, and F1 for prediction using provided threshold.
  '''
  binary_voxel_batch = copy.deepcopy(y_pred)
  binary_voxel_batch[y_pred >= threshold] = 1
  binary_voxel_batch[y_pred < threshold] = 0
  TP = float(np.sum(binary_voxel_batch[y_true == 1] == 1))
  FN = float(np.sum(binary_voxel_batch[y_true == 1] == 0))
  FP = float(np.sum(binary_voxel_batch[y_true == 0] == 1))
  TN = float(np.sum(binary_voxel_batch[y_true == 0] == 0))

  precision = TP / (TP + FP)
  recall = TP / (TP + FN)

  if (precision + recall) == 0.0:
    return 0.0
    
  f1 = 2 * ((precision * recall) / (precision + recall))
  return f1

def get_f1(dataset, model, threshold, i=None):
  '''
  Calculate the average f1 for the given model.
  '''
  accumulate_f1 = 0.0
  elems = 0

  for element in dataset:
    input_tensor = tf.cast(element[0], dtype=tf.float32)
    y = model(input_tensor)
    f1 = f1_prediction(element[0].numpy(), y.numpy(), threshold)
    accumulate_f1 += f1
    elems += 1

    if i is not None and elems == i:
      break

  return accumulate_f1 / elems

def get_loss(dataset, model, i=None):
  '''
  Calculate the current loss for the given model.
  '''
  accumulate_loss = 0.0
  elems = 0
  
  for element in dataset:
    input_tensor = tf.cast(element[0], dtype=tf.float32)
    y = model(input_tensor)
    loss = lambda_binary_crossentropy(input_tensor, y)
    accumulate_loss += loss
    elems += 1

    if i is not None and elems == i:
      break

  return accumulate_loss / elems

def get_latent_loss(dataset, partial_encoder, full_encoder, i=None):
  '''
  Calculate MSE Loss for the encoder to latent space. Note, dataset should return
  both latent and full.
  '''
  accumulate_loss = 0.0
  elems = 0

  for element in dataset:
    partial_tensor = tf.cast(element[0][:,0,:,:,:,:], dtype=tf.float32)
    full_tensor = tf.cast(element[0][:,1,:,:,:,:], dtype=tf.float32)
    y_true = full_encoder(full_tensor)
    y_pred = partial_encoder(partial_tensor)
    loss = mean_squared_error(y_true, y_pred)
    accumulate_loss += tf.reduce_sum(loss)
    elems += 1

    if i is not None and elems == i:
      break

  return accumulate_loss / elems
