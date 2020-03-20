input_data = tf.Variable(np.random.rand(10,9,9,3),dtype=np.float32)
depthwise_filter = tf.Variable(np.random.rand(2,2,3,5),dtype=np.float32)
pointerwise_filter = tf.Variable(np.random.rand(1,1,15,20),dtype=np.float32)
#out_channels >= channel_multiplier * in_channels
y =tf.nn.separable_conv2d(input_data, depthwise_filter, pointerwise_filter, strides = [1,1,1,1], padding='SAME')
y.shape