
import tensorflow as tf
import time
import matplotlib.pyplot as plt
import cifar_reader


batch_size = 100
step = 0
train_iter = 50000
display_step = 10


# 初始化单个卷积核上的权重
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


# 初始化单个卷积核上的偏置值
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


# 输入特征x，用卷积核W进行卷积运算，strides为卷积核移动步长，
# padding表示是否需要补齐边缘像素使输出图像大小不变
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


# 对x进行最大池化操作，ksize进行池化的范围，
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
#第一个参数value：需要池化的输入，一般池化层接在卷积层后面，所以输入通常是feature map，依然是[batch, height, width, channels]这样的shape
#第二个参数ksize：池化窗口的大小，取一个四维向量，一般是[1, height, width, 1]，因为我们不想在batch和channels上做池化，所以这两个维度设为了1
#第三个参数strides：和卷积类似，窗口在每一个维度上滑动的步长，一般也是[1, stride,stride, 1]
#第四个参数padding：和卷积类似，可以取’VALID’ 或者’SAME’
    
sess = tf.InteractiveSession()
# 声明输入图片数据，类别
x = tf.placeholder('float32', [None, 1024])
y_= tf.placeholder(dtype=tf.float32, shape=[None, 10])
input_x = tf.reshape(x, [-1, 32, 32, 3])
keep_prob = tf.placeholder(tf.float32)
is_traing = tf.placeholder(tf.bool)


W_conv1 = weight_variable([5, 5, 3, 64])
b_conv1 = bias_variable([64])
h_conv1 = tf.nn.relu(conv2d(input_x, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)







W_conv2 = weight_variable([5, 5, 64, 128])
b_conv2 = bias_variable([128])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

W_conv3 = weight_variable([5, 5, 128, 256])
b_conv3 = bias_variable([256])
h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
h_pool3 = max_pool_2x2(h_conv3)

W_fc1 = weight_variable([4 * 4 * 256, 1024])
# 偏置值
b_fc1 = bias_variable([1024])
# 将卷积的产出展开
h_pool2_flat = tf.reshape(h_pool3, [-1, 8 * 8 * 64])
# 神经网络计算，并添加relu激活函数
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

W_fc2 = weight_variable([1024, 128])
b_fc2 = bias_variable([128])
h_fc2 = tf.nn.relu(tf.matmul(h_fc1, W_fc2) + b_fc2)

W_fc3 = weight_variable([128, 10])
b_fc3 = bias_variable([10])
y_conv = tf.nn.softmax(tf.matmul(h_fc2, W_fc3) + b_fc3)
# 代价函数
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_conv, labels=y_))
optimizer = tf.train.AdamOptimizer(0.01).minimize(cost)
correct_pred = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
# 所有变量进行初始化
sess.run(tf.initialize_all_variables())

# 获取mnist数据
dr = cifar_reader.Cifar10DataReader(cifar_folder="./cifar-10-batches-py/")
c = []
# 进行训练
start_time = time.time()

 
saver = tf.train.Saver()
 
while step * batch_size < train_iter: 
    step += 1
    batch_xs, batch_ys = dr.next_train_data(batch_size)
    # 获取批数据,计算精度, 损失值
    opt, acc, loss = sess.run([optimizer, accuracy, cost],
                              feed_dict={input_x: batch_xs, y_: batch_ys, keep_prob: 0.6, is_traing: True})
    if step % display_step == 0:
        print ("Iter " + str(step*batch_size) + ", Minibatch Loss= " + "{:.6f}".format(loss) + ", Training Accuracy= " + "{:.5f}".format(acc))
print ("Optimization Finished!")
 
    # 计算测试精度
num_examples = 10000
d, l = dr.next_test_data(num_examples)
print ("Testing Accuracy:", sess.run(accuracy, feed_dict={input_x: d, y_: l, keep_prob: 1.0, is_traing: True}))
saver.save(sess, "model_tmp/cifar10_demo.ckpt")



sess.close()
plt.plot(c)
plt.tight_layout()
