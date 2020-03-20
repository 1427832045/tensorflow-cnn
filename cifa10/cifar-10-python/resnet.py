import tensorflow as tf
import cifar_reader
import matplotlib.pyplot as plt

batch_size = 100
step = 0
train_iter = 500000
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
    return tf.nn.max_pool(x, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')



def identity_block(X_input, kernel_size, in_filter, out_filters, stage, block):
        #x2, 3, 256, [64, 64, 256], stage=2, block='b'
        """
        Implementation of the identity block as defined in Figure 3
        Arguments:
        X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
        kernel_size -- integer, specifying the shape of the middle CONV's window for the main path
        filters -- python list of integers, defining the number of filters in the CONV layers of the main path
        stage -- integer, used to name the layers, depending on their position in the network
        block -- string/character, used to name the layers, depending on their position in the network
        training -- train or test
        Returns:
        X -- output of the identity block, tensor of shape (n_H, n_W, n_C)
        """
 
        # defining name basis
        block_name = 'res' + str(stage) + block
        f1, f2, f3 = out_filters  #64x64x256
        with tf.variable_scope(block_name):
            X_shortcut = X_input
 
            #first
            #1x1x256 64个卷积核
            W_conv1 = weight_variable([1, 1, in_filter, f1])
            X = tf.nn.conv2d(X_input, W_conv1, strides=[1, 1, 1, 1], padding='SAME')
            b_conv1 = bias_variable([f1])
            X = tf.nn.relu(X+ b_conv1)
 
            #second
            #3x3x64 64个卷积核
            W_conv2 = weight_variable([kernel_size, kernel_size, f1, f2])
            X = tf.nn.conv2d(X, W_conv2, strides=[1, 1, 1, 1], padding='SAME')
            b_conv2 = bias_variable([f2])
            X = tf.nn.relu(X+ b_conv2)
 
            #third
            #1x1x64 256个卷积核
            W_conv3 = weight_variable([1, 1, f2, f3])
            X = tf.nn.conv2d(X, W_conv3, strides=[1, 1, 1, 1], padding='SAME')
            b_conv3 = bias_variable([f3])
            X = tf.nn.relu(X+ b_conv3)
            #final step
            add = tf.add(X, X_shortcut)
            #b_conv_fin = bias_variable([f3])
            add_result = tf.nn.relu(add)
 
        return add_result
 
 
#这里定义conv_block模块，由于该模块定义时输入和输出尺度不同，故需要进行卷积操作来改变尺度，从而得以相加
def convolutional_block( X_input, kernel_size, in_filter,
                            out_filters, stage, block, stride=1):
        #X_input=x1, kernel_size=3, in_filter=64,  out_filters=[64, 64, 256], stage=2, block='a', stride=1
        """
        Implementation of the convolutional block as defined in Figure 4
        Arguments:
        X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
        kernel_size -- integer, specifying the shape of the middle CONV's window for the main path
        filters -- python list of integers, defining the number of filters in the CONV layers of the main path
        stage -- integer, used to name the layers, depending on their position in the network
        block -- string/character, used to name the layers, depending on their position in the network
        training -- train or test
        stride -- Integer, specifying the stride to be used
        Returns:
        X -- output of the convolutional block, tensor of shape (n_H, n_W, n_C)
        """
 
        # defining name basis
        block_name = 'res' + str(stage) + block
        with tf.variable_scope(block_name):
            f1, f2, f3 = out_filters
 
            x_shortcut = X_input#输入为14x14x64
            #first
            #卷积核为1x1x64,64个卷积核
            #输入为14x14x64输出为7x7x64
            W_conv1 = weight_variable([1, 1, in_filter, f1])
            X = tf.nn.conv2d(X_input, W_conv1,strides=[1, stride, stride, 1],padding='SAME')
            b_conv1 = bias_variable([f1])
            X = tf.nn.relu(X + b_conv1)
 
            #second
            #卷积核为3x3x64 64个卷积核
            #输入为7x7x64输出为7x7x64
            W_conv2 =weight_variable([kernel_size, kernel_size, f1, f2])
            X = tf.nn.conv2d(X, W_conv2, strides=[1,1,1,1], padding='SAME')
            b_conv2 = bias_variable([f2])
            X = tf.nn.relu(X+b_conv2)
 
            #third
            #卷积核为1x1x64,256个卷积核
            #输入为7x7x64输出为7x7x256
            W_conv3 = weight_variable([1,1, f2,f3])
            X = tf.nn.conv2d(X, W_conv3, strides=[1, 1, 1,1], padding='SAME')
            b_conv3 = bias_variable([f3])
            X = tf.nn.relu(X+b_conv3)
            #shortcut path
            #输入为14x14x64输出为7x7x256
            W_shortcut =weight_variable([1, 1, in_filter, f3])
            x_shortcut = tf.nn.conv2d(x_shortcut, W_shortcut, strides=[1, stride, stride, 1], padding='VALID')
 
            #final
            add = tf.add(x_shortcut, X)
            #建立最后融合的权重
            #b_conv_fin = bias_variable([f3])
            add_result = tf.nn.relu(add)
            #最后为7x7x256
 
 
        return add_result

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
h_conv1 = tf.nn.relu(conv2d1(input_x, W_conv1) + b_conv1)
print(h_conv1)
lrn1 = tf.nn.lrn(h_conv1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='lrn1')
print(lrn1)
pool_1 = max_pool_2x2(lrn1)
print(pool_1)

x2 = convolutional_block(X_input=pool_1, kernel_size=3, in_filter=64,  out_filters=[64, 64, 256], stage=2, block='a', stride=1)
print(x2)
x3 = identity_block(x2, 3, 256, [64, 64, 256], stage=2, block='b' )
print(x3)
x4 = identity_block(x3, 3, 256, [64, 64, 256], stage=2, block='c')
print(x4)
pool_2 = max_pool_2x2(x4)
print(pool_2)

 
#fc1
dense_tmp = tf.reshape(pool_2, shape=[-1, 8*8*256])
print(dense_tmp)
 
fc1 = tf.Variable(tf.truncated_normal(shape=[8*8
                                             *256, 1024], stddev=0.04))
 
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
