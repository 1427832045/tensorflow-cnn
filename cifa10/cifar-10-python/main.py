import tensorflow as tf
import cifar_reader
 
batch_size = 100
step = 0
train_iter = 50000
display_step = 10
 
# for key in data:
#     print(key)
input_x = tf.placeholder(dtype=tf.float32, shape=[None, 32, 32, 3])
y = tf.placeholder(dtype=tf.float32, shape=[None, 10])
keep_prob = tf.placeholder(tf.float32)
is_traing = tf.placeholder(tf.bool)
 
####conv1
W1 = tf.Variable(tf.truncated_normal([3, 3, 3, 64], dtype=tf.float32, stddev=5e-2))
conv_1 = tf.nn.conv2d(input_x, W1, strides=(1, 1, 1, 1), padding="VALID")
print(conv_1)
 
bn1 = tf.layers.batch_normalization(conv_1, training=is_traing)
 
relu_1 = tf.nn.relu(bn1)
print(relu_1)
 
pool_1 = tf.nn.max_pool(relu_1, strides=[1, 2, 2, 1], padding="VALID", ksize=[1, 3, 3, 1])
print(pool_1)
 
####conv2
W2 = tf.Variable(tf.truncated_normal(shape=[3, 3, 64, 128], dtype=tf.float32, stddev=5e-2))
conv_2 = tf.nn.conv2d(pool_1, W2, strides=[1, 1, 1, 1], padding="SAME")
print(conv_2)
 
bn2 = tf.layers.batch_normalization(conv_2, training=is_traing)
 
relu_2 = tf.nn.relu(bn2)
print(relu_2)
 
pool_2 = tf.nn.max_pool(relu_2, strides=[1, 2, 2, 1], ksize=[1, 3, 3, 1], padding="VALID")
print(pool_2)
 
####conv3
W3 = tf.Variable(tf.truncated_normal(shape=[3, 3, 128, 256], dtype=tf.float32, stddev=1e-1))
conv_3 = tf.nn.conv2d(pool_2, W3, strides=[1, 1, 1, 1], padding="SAME")
print(conv_3)
 
bn3 = tf.layers.batch_normalization(conv_3, training=is_traing)
 
relu_3 = tf.nn.relu(bn3)
print(relu_3)
 
pool_3 = tf.nn.max_pool(relu_3, strides=[1, 2, 2, 1], ksize=[1, 3, 3, 1], padding="VALID")
print(pool_3)
 
#fc1
dense_tmp = tf.reshape(pool_3, shape=[-1, 2*2*256])
print(dense_tmp)
 
fc1 = tf.Variable(tf.truncated_normal(shape=[2*2*256, 1024], stddev=0.04))
 
bn_fc1 = tf.layers.batch_normalization(tf.matmul(dense_tmp, fc1), training=is_traing)
 
dense1 = tf.nn.relu(bn_fc1)
dropout1 = tf.nn.dropout(dense1, keep_prob)
 
#fc2
fc2 = tf.Variable(tf.truncated_normal(shape=[1024, 10], stddev=0.04))
out = tf.matmul(dropout1, fc2)
print(out)
 
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=out, labels=y))
optimizer = tf.train.AdamOptimizer(0.01).minimize(cost)
 
dr = cifar_reader.Cifar10DataReader(cifar_folder="./cifar-10-batches-py/")
 
# 测试网络
correct_pred = tf.equal(tf.argmax(out, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
 
# 初始化所有的共享变量
init = tf.initialize_all_variables()
 
saver = tf.train.Saver()
 
# 开启一个训练
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    # saver.restore(sess, "model_tmp/cifar10_demo.ckpt")
    step = 1
 
    # Keep training until reach max iterations
    while step * batch_size < train_iter:
        step += 1
        batch_xs, batch_ys = dr.next_train_data(batch_size)
        # 获取批数据,计算精度, 损失值
        opt, acc, loss = sess.run([optimizer, accuracy, cost],
                                  feed_dict={input_x: batch_xs, y: batch_ys, keep_prob: 0.6, is_traing: True})
        if step % display_step == 0:
            print ("Iter " + str(step*batch_size) + ", Minibatch Loss= " + "{:.6f}".format(loss) + ", Training Accuracy= " + "{:.5f}".format(acc))
    print ("Optimization Finished!")
 
    # 计算测试精度
    num_examples = 10000
    d, l = dr.next_test_data(num_examples)
    print ("Testing Accuracy:", sess.run(accuracy, feed_dict={input_x: d, y: l, keep_prob: 1.0, is_traing: True}))
    saver.save(sess, "model_tmp/cifar10_demo.ckpt")
