import numpy as np
import time
import os
import tensorflow as tf

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

def get_voxel_ae(load_weights_file=None, verbose=False):
    '''
    Build and return Voxel Model.
    '''

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
    if verbose:
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
    if verbose:
        decoder.summary()

    ################################# AUTOENCODER ###############################
    reconstruction = decoder(encoder(voxel_input))
    ae = Model(inputs=voxel_input, outputs=reconstruction)

    if verbose:
        ae.summary()

    # If provided, load weights.
    if load_weights_file is not None:
        ae.load_weights(load_weights_file)
        
    return ae
