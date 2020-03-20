import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import _pickle as pickle
import time

'''返回的dict是一个包含0-9数字的list列表，如[0,2,1,4,6,3,5,...,4,2,5]'''
def unpickle(file):
    with open(file,'rb') as fo:
        dict=pickle.load(fo,encoding='latin1')
    return dict

'''将上面的dict转换成对应的one-hot矩阵，shape=[n_sample,n_class]'''
def onehot(labels):
    n_sample=len(labels) #数据集的数量
    n_class=max(labels)+1 #one_hot分类的数量
    onehot_labels=np.zeros((n_sample,n_class))
    onehot_labels[np.arange(n_sample),labels]=1
    return onehot_labels

#训练数据集
data1=unpickle('cifar-10-batches-py/data_batch_1')
data2=unpickle('cifar-10-batches-py/data_batch_2')
data3=unpickle('cifar-10-batches-py/data_batch_3')
data4=unpickle('cifar-10-batches-py/data_batch_4')
data5=unpickle('cifar-10-batches-py/data_batch_5')

x_train=np.concatenate((data1['data'],data2['data'],data3['data'],data4['data'],data5['data']),axis=0)
y_train=np.concatenate((data1['labels'],data2['labels'],data3['labels'],data4['labels'],data5['labels']),axis=0)
#转换格式
y_train=onehot(y_train)

#测试集
test=unpickle('cifar-10-batches-py/test_batch')
x_test=test['data'][:5000,:]
y_test=onehot(test['labels'])[:5000,:]

print('Training dataset shape:',x_train.shape)
print('Training labels shape:',y_train.shape)
print('Testing dataset shape:',x_test.shape)
print('Testing labels shape:',y_test.shape)

with tf.device('/gpu:0'):
    #模型参数
    lr=1e-3
    training_iters=200
    batch_size=50
    display_step=5
    n_features=3072 #32*32*3
    n_classes=10
    n_fc1=384
    n_fc2=192

    #构建模型
    x=tf.placeholder(tf.float32,[None,n_features])
    y=tf.placeholder(tf.float32,[None,n_classes])

    W_conv={
        'conv1':tf.Variable(tf.truncated_normal([5,5,3,32],stddev=0.0001)),
        'conv2':tf.Variable(tf.truncated_normal([5,5,32,64],stddev=0.01)),
        'fc1':tf.Variable(tf.truncated_normal([8*8*64,n_fc1],stddev=0.01)),
        'fc2':tf.Variable(tf.truncated_normal([n_fc1,n_fc2],stddev=0.1)),
        'fc3':tf.Variable(tf.truncated_normal([n_fc2,n_classes],stddev=0.1))
    }

    b_conv={
        'conv1':tf.Variable(tf.constant(0.0,dtype=tf.float32,shape=[32])),
        'conv2':tf.Variable(tf.constant(0.1,dtype=tf.float32,shape=[64])),
        'fc1':tf.Variable(tf.constant(0.1,dtype=tf.float32,shape=[n_fc1])),
        'fc2':tf.Variable(tf.constant(0.1,dtype=tf.float32,shape=[n_fc2])),
        'fc3':tf.Variable(tf.constant(0.0,dtype=tf.float32,shape=[n_classes]))
    }

    x_image=tf.reshape(x,[-1,32,32,3])
    #卷积层1
    conv1=tf.nn.conv2d(x_image,W_conv['conv1'],strides=[1,1,1,1],padding='SAME')
    conv1=tf.nn.bias_add(conv1,b_conv['conv1'])
    conv1=tf.nn.relu(conv1)
    #池化层1
    pool1=tf.nn.avg_pool(conv1,ksize=[1,3,3,1],strides=[1,2,2,1],padding='SAME')
    #LRN层
    norm1=tf.nn.lrn(pool1,4,bias=1.0,alpha=0.001/9,beta=0.75)
    #卷积层2
    conv2=tf.nn.conv2d(norm1,W_conv['conv2'],strides=[1,1,1,1],padding='SAME')
    conv2=tf.nn.bias_add(conv2,b_conv['conv2'])
    conv2=tf.nn.relu(conv2)
    # LRN层(归一化层)
    norm2=tf.nn.lrn(conv2,4,bias=1.0,alpha=0.001/9.0,beta=0.75)
    #池化层
    pool2=tf.nn.avg_pool(norm2,ksize=[1,3,3,1],strides=[1,2,2,1],padding='SAME')
    reshape=tf.reshape(pool2,[-1,8*8*64])
    #全连接层1
    fc1=tf.add(tf.matmul(reshape,W_conv['fc1']),b_conv['fc1'])
    fc1=tf.nn.relu(fc1)
    #全连接层2
    fc2=tf.add(tf.matmul(fc1,W_conv['fc2']),b_conv['fc2'])
    fc2=tf.nn.relu(fc2)
    #全连接层3，即分类层
    fc3=tf.nn.softmax(tf.add(tf.matmul(fc2,W_conv['fc3']),b_conv['fc3']))

    #定义损失函数
    loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=fc3,labels=y))
    optimizer=tf.train.GradientDescentOptimizer(learning_rate=lr).minimize(loss)
    #评估模型
    correct_pred=tf.equal(tf.argmax(fc3,1),tf.argmax(y,1))
    accuracy=tf.reduce_mean(tf.cast(correct_pred,tf.float32))

    init=tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    c=[]
    total_batch=int(x_train.shape[0]/batch_size)
    start_time=time.time()
    for i in range(200):
        acc=[]
        for batch in range(total_batch):
            batch_x=x_train[batch*batch_size:(batch+1)*batch_size,:]
            batch_y=y_train[batch*batch_size:(batch+1)*batch_size,:]
            sess.run(optimizer,feed_dict={x:batch_x,y:batch_y})
            acc=sess.run(accuracy,feed_dict={x:batch_x,y:batch_y})
        print(acc)
        c.append(acc)
        end_time=time.time()
        print('time:',end_time-start_time)
        start_time=end_time

    #Test
    test_acc=sess.run(accuracy,feed_dict={x:x_test,y:y_test})
    print("Testing Accuarcy:",test_acc)
    plt.plot(c)
    plt.xlabel('Iter')
    plt.ylabel('Cost')
    plt.title('lr=%f,ti=%d,bs=%d,acc=%f'%(lr,training_iters,batch_size,test_acc))
    plt.tight_layout()
    plt.savefig('cnn-tf-cifar-%s.png'%test_acc,dpi=200)

