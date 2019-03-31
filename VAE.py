import numpy as np
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers import Flatten, Conv3D, Dense
from keras.engine.input_layer import Input
from keras.losses import logcosh



######################### LOSS FUNCTION ###################################
# TODO: LOSS FUNCTION FROM PAPER

########################## ENCODER NETWORK #################################

# Create model
# All layers use 'elu' activation
model = Sequential()

# First convolutional layer
# Glorot initialization is deafault (glorot_uniform) - Keras has glorot_normal 
# and glorot_uniform, paper is unclear which should be used
model.add(Conv3D(8, kernel_size=3, activation='elu', padding='valid', data_format='channels_last', input_shape=(32, 32, 32, 1))) #input_shape=(32, 32, 32, 1)
model.add(BatchNormalization())

# Second convolutional layer, downsampling through strided convolutions 
# (default striding is 1x1x1, so up this to 2x2x2)
model.add(Conv3D(16, kernel_size=3, activation='elu', padding='valid', strides=(2, 2, 2))) #strides=(2, 2, 2)
model.add(BatchNormalization())

# Third convolutional layer
model.add(Conv3D(32, kernel_size=3, activation='elu', padding='valid'))
model.add(BatchNormalization())

# Fourth convolutional layer, downsampling through strided convolutions
model.add(Conv3D(64, kernel_size=3, activation='elu', padding='valid', strides=(2, 2, 2)))
model.add(BatchNormalization())

# Fully connected layer and latent layer
# Fully connected: Dense layer that takes in flattened output of previous layer 
# Latent layer: output of fully connected layer is 100 dimensional latent space
model.add(Flatten())
# print(model.output_shape)
model.add(Dense(100, use_bias=True, activation='elu'))
# model.add(BatchNormalization())


# Stats for the encoder network 
model.compile(optimizer='adam', loss='logcosh', metrics=['accuracy'])
model.summary()


# Testing encoder
# Generating 100 random 32x32 arrays as dummy data 
dummy_data = np.random.rand(100, 32, 32, 32, 1)
# Labels don't matter for us, we only care about the model's ouptut, output must be in the shape of final layer!!!
dumb_labels = np.random.rand(100, 100)
  
model.fit(dummy_data, dumb_labels, shuffle=True, batch_size=100)

