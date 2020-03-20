
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import time
import matplotlib.pyplot as plt
dropout = 0.8 
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
def global_ave_pool(x, x_shape):
  'global average pooling the input x'
  'x       :input feature map [1,x,y,1]'
  'x_shape :should be [1,x,y,1]'
  return tf.nn.avg_pool(x, x_shape,[1,2,2,1], 'VALID') 
    


sess = tf.InteractiveSession()
# 声明输入图片数据，类别
x = tf.placeholder('float32', [None, 784])
y_ = tf.placeholder('float32', [None, 10])
# 输入图片数据转化
keep_prob = tf.placeholder(tf.float32)
x_image = tf.reshape(x, [-1, 28, 28, 1])

W_conv1 = weight_variable([3, 3, 1, 128])
b_conv1 = bias_variable([128])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)

W_conv2 = weight_variable([1, 1, 128, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_conv1, W_conv2) + b_conv2)

W_conv3 = weight_variable([1, 1, 64, 64])
b_conv3 = bias_variable([64])
h_conv3 = tf.nn.relu(conv2d(h_conv2, W_conv3) + b_conv3)
h_pool3 = max_pool_2x2(h_conv3)
#h_drop1 = tf.nn.dropout(h_pool3, 0.5)

W_conv4 = weight_variable([3, 3, 64, 128])
b_conv4 = bias_variable([128])
h_conv4 = tf.nn.relu(conv2d(h_pool3, W_conv4) + b_conv4)

W_conv5 = weight_variable([1, 1, 128, 128])
b_conv5 = bias_variable([128])
h_conv5 = tf.nn.relu(conv2d(h_conv4, W_conv5) + b_conv5)

W_conv6 = weight_variable([1, 1, 128, 128])
b_conv6 = bias_variable([128])
h_conv6 = tf.nn.relu(conv2d(h_conv5, W_conv6) + b_conv6)
h_pool6 = max_pool_2x2(h_conv6)
h_drop2 = tf.nn.dropout(h_pool6 , 0.5)


W_conv7 = weight_variable([3, 3, 128, 128])
b_conv7 = bias_variable([128])
h_conv7 = tf.nn.relu(conv2d(h_drop2, W_conv7) + b_conv7)

W_conv8 = weight_variable([1, 1, 128, 128])
b_conv8 = bias_variable([128])
h_conv8 = tf.nn.relu(conv2d(h_conv7, W_conv8) + b_conv8)

W_conv9 = weight_variable([1, 1, 128, 10])
b_conv9 = bias_variable([10])
h_conv9 = tf.nn.relu(conv2d(h_conv8, W_conv9) + b_conv9)

h_drop3 = tf.nn.dropout(h_conv9 , 0.5)
ave_vec  = global_ave_pool(h_drop3, [1,7,7,1])
h        = tf.reshape(ave_vec, [-1, 10])
y_conv   = h

#h_drop3 = tf.nn.dropout(h_conv9 , 0.5)



#ave_vec  = global_ave_pool(h_drop3, [1,7,7,1])
#h        = tf.reshape(ave_vec, [-1, 10])
#y_conv   = h
#W_fc1 = weight_variable([7 * 7 * 64, 1024])
# 偏置值
#b_fc1 = bias_variable([1024])
# 将卷积的产出展开
#h_pool2_flat = tf.reshape(h_conv9, [-1, 7 * 7 * 64])
# 神经网络计算，并添加relu激活函数
#h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

#h_drop3 = tf.nn.dropout(h_fc1 , 0.5)
#W_fc2 = weight_variable([1024, 128])
#b_fc2 = bias_variable([128])
#h_fc2 = tf.nn.relu(tf.matmul(h_drop3, W_fc2) + b_fc2)

#W_fc3 = weight_variable([128, 10])
#b_fc3 = bias_variable([10])
#y_conv = tf.nn.softmax(tf.matmul(h_fc2, W_fc3) + b_fc3)





# 代价函数
cross_entropy      = tf.reduce_mean(
                     tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
train_step         = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
#train_step         = tf.train.GradientDescentOptimizer(0.0001).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy           = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# 所有变量进行初始化
sess.run(tf.initialize_all_variables())

# 获取mnist数据
mnist_data_set = input_data.read_data_sets('MNIST_data', one_hot=True)
c = []
# 进行训练
start_time = time.time()
for i in range(2000):
    # 获取训练数据
    batch_xs, batch_ys = mnist_data_set.train.next_batch(200)

    # 每迭代10个 batch，对当前训练数据进行测试，输出当前预测准确率
    if i % 5 == 0:
        train_accuracy = accuracy.eval(feed_dict={x: batch_xs, y_: batch_ys})
        c.append(train_accuracy)
        print("step %d, training accuracy %g" % (i, train_accuracy))
        # 计算间隔时间
        end_time = time.time()
        print('time: ', (end_time - start_time))
        start_time = end_time
    # 训练数据
    train_step.run(feed_dict={x: batch_xs, y_: batch_ys})


print("test accuracy %g"%accuracy.eval(feed_dict={
    x: mnist_data_set.test.images, y_: mnist_data_set.test.labels, keep_prob: 1.0}))  


sess.close()
plt.plot(c)
plt.tight_layout()
