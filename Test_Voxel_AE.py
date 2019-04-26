else:
  # Load model from checkpoint.
  ae.load_weights(model_checkpoint_file)

  # Get tf.data.Dataset of our voxels.
  train_dataset, test_dataset = get_voxel_dataset(batch_size=1)

  for train_x in test_dataset:
    # View input.
    input_tensor = tf.cast(train_x[0], dtype=tf.float32)
    input_npy = input_tensor.numpy().reshape((32,32,32))
    # input_npy = (input_npy + 1) // 3
    input_sparse = convert_to_sparse_voxel_grid(input_npy)
    print "True:"
    plot_voxel(input_sparse, voxel_res=(32,32,32))

    # Generate and view reconstruction.
    y = ae(input_tensor)
    y = y.numpy().reshape((32,32,32))
    print y
    y_sparse = convert_to_sparse_voxel_grid(y)
    print "Predict:"
    plot_voxel(y_sparse, voxel_res=(32,32,32))
