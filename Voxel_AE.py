import numpy as np
from keras.models import Model, Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers import Flatten, Conv3D, Dense, Conv1D, Input, Reshape, Conv3DTranspose
from keras.engine.input_layer import Input
from keras.losses import logcosh
from keras.regularizers import l2
from keras import backend as K


################### MODIFIED BINARY CROSS ENTROPY LOSS FUNCTION ############
# Binary cross entropy but with a lambda parameter that
# encourages false positives and discourage false negatives (because of the high
# amounts of blank voxels, without this term the model would output empty voxels)
def lambda_binary_crossentropy(y_true, y_pred):
  output = K.clip(y_pred, 0.1, 1)
  binary_entr = -0.97 * y_true * K.log(output) - (0.03) * (1-y_true) * K.log(1-output)

  # getting tensor values into scalar
  loss = K.sum(binary_entr, axis=1)
  scalar_loss = K.mean(loss)

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

# Train using custom loss, and adam optimizer
ae.compile(optimizer='adam', loss=lambda_binary_crossentropy)
