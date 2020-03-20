import tensorflow as tf
import cifar_reader
import matplotlib.pyplot as plt

batch_size = 100
step = 0
train_iter = 300000
display_step = 10
 

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


# 初始化单个卷积核上的偏置值
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


# 输入特征x，用卷积核W进行卷积运算，strides为卷积核移动步长，
# padding表示是否需要补齐边缘像素使输出图像大小不变
def conv2d1(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def conv2d2(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='VALID')


# 对x进行最大池化操作，ksize进行池化的范围，
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID')

# for key in data:
#     print(key)
    
sess = tf.InteractiveSession()
input_x = tf.placeholder(dtype=tf.float32, shape=[None, 32, 32, 3])
y = tf.placeholder(dtype=tf.float32, shape=[None, 10])
keep_prob = tf.placeholder(tf.float32)
is_traing = tf.placeholder(tf.bool)
 
####conv1
W_conv1 = weight_variable([3, 3, 3, 64])
b_conv1 = bias_variable([64])
h_conv1 = tf.nn.relu(conv2d2(input_x, W_conv1) + b_conv1)
print(h_conv1)
 
pool_1 = max_pool_2x2(h_conv1)
print(pool_1)
 
####conv2
W_conv2 = weight_variable([3, 3, 64, 128])
b_conv2 = bias_variable([128])
h_conv2 = tf.nn.relu(conv2d2(pool_1, W_conv2) + b_conv2)
print(h_conv2)

 
pool_2 = max_pool_2x2(h_conv2)
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
c = []
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
        c.append(acc)
        if step % display_step == 0:
            print ("Iter " + str(step*batch_size) + ", Minibatch Loss= " + "{:.6f}".format(loss) + ", Training Accuracy= " + "{:.5f}".format(acc))
    print ("Optimization Finished!")
 
    # 计算测试精度
    num_examples = 10000
    d, l = dr.next_test_data(num_examples)
    print ("Testing Accuracy:", sess.run(accuracy, feed_dict={input_x: d, y: l, keep_prob: 1.0, is_traing: True}))
    saver.save(sess, "model_tmp/cifar10_demo.ckpt")


sess.close()
plt.plot(c)
plt.tight_layout()
