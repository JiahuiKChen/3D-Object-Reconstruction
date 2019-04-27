import numpy as np
import argparse

from Train_Voxel_AE import train_voxel_ae
from Partial_Latent import train_partial_to_full

parser = argparse.ArgumentParser(description='Run reconstruction codes.')
parser.add_argument('--model_name', '-m', help='Name for model.', required=True)
parser.add_argument('--function', '-f', help='Function to call [train, partial_train, test]', required=True)
parser.add_argument('--epochs', '-e', help='Epochs to train for.', default=100, type=int)
parser.add_argument('--batch_size', '-b', help='Batch size for training.', default=32, type=int)
parser.add_argument('--load_model_file', '-l', help='File to load from.', default=None)
parser.add_argument('--verbose', '-v', action='store_true')
args = parser.parse_args()

if args.function == 'train':
    train_voxel_ae(args.model_name, args.epochs, args.batch_size, args.load_model_file, args.verbose)
elif args.function == 'partial_train':
    train_partial_to_full(args.model_name, args.epochs, args.batch_size, args.load_model_file, args.verbose)
