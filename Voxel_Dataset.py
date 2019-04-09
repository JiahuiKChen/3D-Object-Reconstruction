import tensorflow as tf
import numpy as np

from os import listdir
from os.path import join, isfile

def get_voxel_dataset(batch_size=64):
    '''
    Setup our dataset object for our voxelizations.
    '''

    data_path = '/dataspace/DexnetVoxels'

    # Specify which subfolders in the dexnet dataset to include.
    subfolders = [
        'amazon_picking_challenge',
        'Cat50_ModelDatabase',
        'NTU3D',
        'SHREC14LSGTB',
        'autodesk',
        'KIT',
        'PrincetonShapeBenchmark',
        # 'YCB', # Leave out YCB for testing.
        'BigBIRD',
        'ModelNet40'
    ]

    # Files holds the files as: subfolder/filename.npy.
    files = []
    for subfolder in subfolders:
        print subfolder
        subfolder_path = join(data_path, subfolder)
        for f in listdir(subfolder_path):
            file_path = join(subfolder_path, f)
            if isfile(file_path):
                files.append(file_path)

    def _load_voxel(filename):
        voxel_data = np.load(filename.numpy())
        return (voxel_data, voxel_data)

    # Create dataset as file names. These are mapped when dataset
    # is queried for next batch via _load_voxel to the full
    # voxel representation.
    dataset = tf.data.Dataset.from_tensor_slices(files)
    dataset = dataset.map(
        lambda filename: tuple(tf.py_function(
            _load_voxel, [filename], [tf.float64, tf.float64])))
    dataset = dataset.repeat()
    dataset = dataset.batch(batch_size)

    return dataset, len(files) // batch_size

if __name__ == '__main__':
    dataset, n_ = get_voxel_dataset(batch_size=1)
    iterator = dataset.make_one_shot_iterator()
    next_element = iterator.get_next()

    with tf.Session() as sess:
        data = sess.run(next_element)
        print(data)