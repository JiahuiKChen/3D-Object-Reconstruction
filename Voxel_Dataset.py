import tensorflow as tf
import numpy as np
import random

from os import listdir
from os.path import join, isfile

import pdb

def make_dataset(files, batch_size, down_sample):
    '''
    Create dataset from the provided fiels.
    '''

    # returns 1 voxel in the form of a tensor
    def _load_voxel(filename):
        voxels = np.load(filename.numpy())
        
        # Downsample if asked - return both.
        if down_sample:
            down_voxels = np.array(voxels)
            down_voxels[:,16:,:] = 0
            voxel_data = tf.reshape(tf.convert_to_tensor(voxels), (32,32,32,1))
            down_voxel_data = tf.reshape(tf.convert_to_tensor(down_voxels), (32,32,32,1))
            return tf.convert_to_tensor([down_voxel_data, voxel_data])
        else:
            voxel_data = tf.convert_to_tensor(voxels)
            return tf.reshape(voxel_data, (32,32,32,1))

    # Create dataset as file names. These are mapped when dataset
    # is queried for next batch via _load_voxel to the full
    # voxel representation.
    dataset = tf.data.Dataset.from_tensor_slices(files)
    # dataset = dataset.shuffle(buffer_size=10000)
    # Converting dataset of voxel file names to dataset of voxels
    dataset = dataset.map(
        lambda filename: tuple(tf.py_function(
            _load_voxel, [filename], [tf.float64])))
    # dataset = dataset.repeat() # Don't need this because of our current eager execution approach.
    dataset = dataset.batch(batch_size)
    #dataset = dataset.prefetch(4)

    return dataset

def get_voxel_dataset(batch_size=64, down_sample=False):
    '''
    Setup our dataset object for our voxelizations.
    '''

    data_path = '/dataspace/DexnetVoxels'

    # TODO: Better way of setting these.
    # Specify which subfolders in the dexnet dataset to include.
    subfolders = [
        # 'amazon_picking_challenge',
        # 'Cat50_ModelDatabase',
        # 'NTU3D',
        # 'SHREC14LSGTB',
        # 'autodesk',
        # 'KIT',
        # 'PrincetonShapeBenchmark',
        # 'YCB', # Leave out YCB for testing.
        # 'BigBIRD',
        # 'ModelNet40',
        'ModelNet40Alternate', # Alternate data augmentation.
    ]

    # Files holds the files as: subfolder/filename.npy. - each file is a voxel
    files = []
    for subfolder in subfolders:
        print subfolder
        subfolder_path = join(data_path, subfolder)
        for f in listdir(subfolder_path):
            file_path = join(subfolder_path, f)
            if isfile(file_path):
                files.append(file_path)

    print "Dataset size: ", len(files)

    # Shuffling file names (so that dataset of voxels is randomized)
    random.Random(42).shuffle(files)

    train_file_size = int(len(files) * 0.9)
    validation_file_size = 320

    #print "Train size: ", train_file_size
    train_files = files[:train_file_size]
    validation_files = files[train_file_size:train_file_size+validation_file_size]
    test_files = files[train_file_size+validation_file_size:]

    # Shuffle training samples.
    random.shuffle(train_files)

    # Create train/test datasets.
    train_dataset = make_dataset(train_files, batch_size, down_sample)
    validation_dataset = make_dataset(validation_files, batch_size, down_sample)
    test_dataset = make_dataset(test_files, batch_size, down_sample)

    return train_dataset, validation_dataset, test_dataset

if __name__ == '__main__':
    dataset, n_ = get_voxel_dataset(batch_size=1)
    iterator = dataset.make_one_shot_iterator()
    next_element = iterator.get_next()

    with tf.Session() as sess:
        data = sess.run(next_element)
        print(data)
