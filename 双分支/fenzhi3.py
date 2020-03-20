
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import time
import matplotlib.pyplot as plt


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


def norm(l_input, lsize=4):
    return tf.nn.lrn(l_input, lsize, bias=1.0, alpha=0.001 / 9.0, beta=0.75)


def global_ave_pool(x, x_shape):
  'global average pooling the input x'
  'x       :input feature map [1,x,y,1]'
  'x_shape :should be [1,x,y,1]'
  return tf.nn.avg_pool(x, x_shape,[1,2,2,1], 'VALID') 


# 对x进行最大池化操作，ksize进行池化的范围，
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
#第一个参数value：需要池化的输入，一般池化层接在卷积层后面，所以输入通常是feature map，依然是[batch, height, width, channels]这样的shape
#第二个参数ksize：池化窗口的大小，取一个四维向量，一般是[1, height, width, 1]，因为我们不想在batch和channels上做池化，所以这两个维度设为了1
#第三个参数strides：和卷积类似，窗口在每一个维度上滑动的步长，一般也是[1, stride,stride, 1]
#第四个参数padding：和卷积类似，可以取’VALID’ 或者’SAME’

def identity_block(X_input):
    W_conva = weight_variable([3, 3, 32, 1])
    Xa = tf.nn.depthwise_conv2d(X_input, W_conva, strides=[1, 1, 1, 1], padding='SAME')
    b_conva = bias_variable([32])
    Xa = tf.nn.relu(Xa+ b_conva)
    norma = norm(Xa, lsize=4)

    W_convb = weight_variable([1, 1, 32, 32])
    Xb = tf.nn.conv2d(X_input, W_convb, strides=[1, 1, 1, 1], padding='SAME')
    b_convb = bias_variable([32])
    Xb = tf.nn.relu(Xb+ b_convb)
    normb = norm(Xb, lsize=4)
    #third
    out = tf.concat((norma, normb),3)
    #final step
    #b_conv_fin = bias_variable([f3])
    add_result = tf.nn.relu(out)
    W_conv3 = weight_variable([1, 1, 64, 32])
    Xc = tf.nn.conv2d(add_result, W_conv3, strides=[1, 1, 1, 1], padding='SAME')
    b_conv3 = bias_variable([32])
    Xc = tf.nn.relu(Xc+ b_conv3)
    normc = norm(Xc, lsize=4)
    return normc

sess = tf.InteractiveSession()
# 声明输入图片数据，类别
x = tf.placeholder('float32', [None, 784])
y_ = tf.placeholder('float32', [None, 10])
# 输入图片数据转化
x_image = tf.reshape(x, [-1, 28, 28, 1])

W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
#norm1 = norm(h_conv1, lsize=4)

Xc = identity_block(h_conv1)


h_pool1 = max_pool_2x2(Xc)







h_drop2 = tf.nn.dropout(h_pool1 , 0.5)
    #depthwish_filter = tf.get_variable(name = '1' , shape=[3,3,1,192], initializer=tf.ones_initializer)




W_conv3 = weight_variable([3, 3, 32, 32])
b_conv3 = bias_variable([32])
h_conv3 = tf.nn.relu(conv2d(h_drop2, W_conv3) + b_conv3)
#norm3= norm(h_conv3, lsize=4)
h_pool2 = max_pool_2x2(h_conv3)

W_conv4 = weight_variable([3, 3, 32, 32])
b_conv4 = bias_variable([32])
h_conv4 = tf.nn.relu(conv2d(h_pool2 , W_conv4) + b_conv4)


W_conv5 = weight_variable([1, 1, 32, 10])
b_conv5 = bias_variable([10])
h_conv5 = tf.nn.relu(conv2d(h_conv4, W_conv5) + b_conv5)

ave_vec  = global_ave_pool(h_conv5, [1,7,7,1])
h        = tf.reshape(ave_vec, [-1, 10])
y_conv   = h


cross_entropy      = tf.reduce_mean(
                     tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
train_step = tf.train.GradientDescentOptimizer(1e-5).minimize(cross_entropy)
#train_step         = tf.train.GradientDescentOptimizer(0.0001).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy           = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# 所有变量进行初始化
sess.run(tf.initialize_all_variables())
# 所有变量进行初始化
sess.run(tf.initialize_all_variables())

# 获取mnist数据
mnist_data_set = input_data.read_data_sets('MNIST_data', one_hot=True)
c = []
# 进行训练
start_time = time.time()
for i in range(5000):
    # 获取训练数据
    batch_xs, batch_ys = mnist_data_set.train.next_batch(200)

    # 每迭代10个 batch，对当前训练数据进行测试，输出当前预测准确率
    if i % 2 == 0:
        train_accuracy = accuracy.eval(feed_dict={x: batch_xs, y_: batch_ys})
        c.append(train_accuracy)
        print("step %d, training accuracy %g" % (i, train_accuracy))
        # 计算间隔时间
        end_time = time.time()
        print('time: ', (end_time - start_time))
        start_time = end_time
    # 训练数据
    train_step.run(feed_dict={x: batch_xs, y_: batch_ys})


batch_xs1, batch_ys1 = mnist_data_set.test.next_batch(1000)
test_accuracy = accuracy.eval(feed_dict={x: batch_xs1, y_: batch_ys1})
print("test accuracy %g" % (test_accuracy))


sess.close()
plt.plot(c)
plt.tight_layout()
