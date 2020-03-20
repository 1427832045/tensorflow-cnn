import os 
import cv2 
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)
sess = tf.InteractiveSession()

def getTrain():
    train=[[],[]] # 指定训练集的格式，一维为输入数据，一维为其标签
    # 读取所有训练图像，作为训练集
    train_root="mnist_train" 
    labels = os.listdir(train_root)
    for label in labels:
        imgpaths = os.listdir(os.path.join(train_root,label))
        for imgname in imgpaths:
            img = cv2.imread(os.path.join(train_root,label,imgname),0)
            array = np.array(img).flatten() # 将二维图像平铺为一维图像
            array=MaxMinNormalization(array)
            train[0].append(array)
            label_ = [0,0,0,0,0,0,0,0,0,0]
            label_[int(label)] = 1
            train[1].append(label_)
    train = shuff(train)
    return train

def getTest():
    test=[[],[]] # 指定训练集的格式，一维为输入数据，一维为其标签
    # 读取所有训练图像，作为训练集
    test_root="mnist_test" 
    labels = os.listdir(test_root)
    for label in labels:
        imgpaths = os.listdir(os.path.join(test_root,label))
        for imgname in imgpaths:
            img = cv2.imread(os.path.join(test_root,label,imgname),0)
            array = np.array(img).flatten() # 将二维图像平铺为一维图像
            array=MaxMinNormalization(array)
            test[0].append(array)
            label_ = [0,0,0,0,0,0,0,0,0,0]
            label_[int(label)] = 1
            test[1].append(label_)
    test = shuff(test)
    return test[0],test[1]

def shuff(data):
    temp=[]
    for i in range(len(data[0])):
        temp.append([data[0][i],data[1][i]])
    import random
    random.shuffle(temp)
    data=[[],[]]
    for tt in temp:
        data[0].append(tt[0])
        data[1].append(tt[1])
    return data

count = 0
def getBatchNum(batch_size,maxNum):
    global count
    if count ==0:
        count=count+batch_size
        return 0,min(batch_size,maxNum)
    else:
        temp = count
        count=count+batch_size
        if min(count,maxNum)==maxNum:
            count=0
            return getBatchNum(batch_size,maxNum)
        return temp,min(count,maxNum)

def MaxMinNormalization(x):
    x = (x - np.min(x)) / (np.max(x) - np.min(x))
    return x

# 1、权重初始化,偏置初始化
# 为了创建这个模型，我们需要创建大量的权重和偏置项
# 为了不在建立模型的时候反复操作，定义两个函数用于初始化
def weight_variable(shape):
    initial = tf.truncated_normal(shape,stddev=0.1)#正太分布的标准差设为0.1
    return tf.Variable(initial)
def bias_variable(shape):
    initial = tf.constant(0.1,shape=shape)
    return tf.Variable(initial)

# 2、卷积层和池化层也是接下来要重复使用的，因此也为它们定义创建函数
# tf.nn.conv2d是Tensorflow中的二维卷积函数，参数x是输入，w是卷积的参数
# strides代表卷积模块移动的步长，都是1代表会不遗漏地划过图片的每一个点，padding代表边界的处理方式
# padding = 'SAME'，表示padding后卷积的图与原图尺寸一致，激活函数relu()
# tf.nn.max_pool是Tensorflow中的最大池化函数，这里使用2 * 2 的最大池化，即将2 * 2 的像素降为1 * 1的像素
# 最大池化会保留原像素块中灰度值最高的那一个像素，即保留最显著的特征，因为希望整体缩小图片尺寸
# ksize：池化窗口的大小，取一个四维向量，一般是[1,height,width,1]
# 因为我们不想再batch和channel上做池化，一般也是[1,stride,stride,1]
def conv2d(x, w):
    return tf.nn.conv2d(x, w, strides=[1,1,1,1],padding='SAME') # 保证输出和输入是同样大小
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1],padding='SAME')

iterNum = 4
batch_size=64
train=getTrain()
test0,test1=getTest()

# 3、参数
# 这里的x,y_并不是特定的值，它们只是一个占位符，可以在TensorFlow运行某一计算时根据该占位符输入具体的值
# 输入图片x是一个2维的浮点数张量，这里分配给它的shape为[None, 784]，784是一张展平的MNIST图片的维度
# None 表示其值的大小不定，在这里作为第1个维度值，用以指代batch的大小，means x 的数量不定
# 输出类别y_也是一个2维张量，其中每一行为一个10维的one_hot向量，用于代表某一MNIST图片的类别
x = tf.placeholder(tf.float32, [None,784], name="x-input")
y_ = tf.placeholder(tf.float32,[None,10]) # 10列

# 4、第一层卷积，它由一个卷积接一个max pooling完成
# 张量形状[5,5,1,32]代表卷积核尺寸为5 * 5，1个颜色通道，32个通道数目
w_conv1 = weight_variable([5,5,1,32])
b_conv1 = bias_variable([32]) # 每个输出通道都有一个对应的偏置量
# 我们把x变成一个4d 向量其第2、第3维对应图片的宽、高，最后一维代表图片的颜色通道数(灰度图的通道数为1，如果是RGB彩色图，则为3)
x_image = tf.reshape(x,[-1,28,28,1])
# 因为只有一个颜色通道，故最终尺寸为[-1，28，28，1]，前面的-1代表样本数量不固定，最后的1代表颜色通道数量
h_conv1 = tf.nn.relu(conv2d(x_image, w_conv1) + b_conv1) # 使用conv2d函数进行卷积操作，非线性处理
h_pool1 = max_pool_2x2(h_conv1)                          # 对卷积的输出结果进行池化操作

# 5、第二个和第一个一样，是为了构建一个更深的网络，把几个类似的堆叠起来
# 第二层中，每个5 * 5 的卷积核会得到64个特征
w_conv2 = weight_variable([5,5,32,64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, w_conv2) + b_conv2)# 输入的是第一层池化的结果
h_pool2 = max_pool_2x2(h_conv2)

# 6、密集连接层
# 图片尺寸减小到7 * 7，加入一个有1024个神经元的全连接层，
# 把池化层输出的张量reshape(此函数可以重新调整矩阵的行、列、维数)成一些向量，加上偏置，然后对其使用Relu激活函数
w_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])
h_pool2_flat = tf.reshape(h_pool2, [-1,7 * 7 * 64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, w_fc1) + b_fc1)

# 7、使用dropout，防止过度拟合
# dropout是在神经网络里面使用的方法，以此来防止过拟合
# 用一个placeholder来代表一个神经元的输出
# tf.nn.dropout操作除了可以屏蔽神经元的输出外，
# 还会自动处理神经元输出值的scale，所以用dropout的时候可以不用考虑scale
keep_prob = tf.placeholder(tf.float32, name="keep_prob")# placeholder是占位符
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# 8、输出层，最后添加一个softmax层
w_fc2 = weight_variable([1024,10])
b_fc2 = bias_variable([10])
y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, w_fc2) + b_fc2, name="y-pred")

# 9、训练和评估模型
# 损失函数是目标类别和预测类别之间的交叉熵
# 参数keep_prob控制dropout比例，然后每100次迭代输出一次日志
cross_entropy = tf.reduce_sum(-tf.reduce_sum(y_ * tf.log(y_conv),reduction_indices=[1]))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
# 预测结果与真实值的一致性，这里产生的是一个bool型的向量
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
# 将bool型转换成float型，然后求平均值，即正确的比例
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
# 初始化所有变量，在2017年3月2号以后,用 tf.global_variables_initializer()替代tf.initialize_all_variables()
sess.run(tf.initialize_all_variables())

# 保存最后一个模型
saver = tf.train.Saver(max_to_keep=1)

for i in range(iterNum):
    for j in range(int(len(train[1])/batch_size)):
        imagesNum=getBatchNum(batch_size,len(train[1]))
        batch = [train[0][imagesNum[0]:imagesNum[1]],train[1][imagesNum[0]:imagesNum[1]]]
        train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
    if i % 2 == 0:
        train_accuracy = accuracy.eval(feed_dict={x: batch[0], y_: batch[1],keep_prob: 1.0})
        print("Step %d ,training accuracy %g" % (i, train_accuracy))
print("test accuracy %f " % accuracy.eval(feed_dict={x: test0, y_:test1, keep_prob: 1.0})) 
# 保存模型于文件夹
saver.save(sess,"save/model")