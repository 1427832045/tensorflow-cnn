import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cPickle
import tensorflow as tf
import sys

def concatenate_kernel_map(kernel, kXSize, kYSize, outx, outy, padding):
  "kernel:  groups of kernel to display [xdim,ydim,1,batch]"
  "kXSize:  kernel x size"
  "kYSize:  kernel y size"
  "outx:    how many kernel patches in output kernel map x direction"
  "outy:    how many kernel patches in putput kernel map y direction"
  "padding: number of padding of zero patches for the last row"
  if padding!=0:
    ZeroPad     = tf.zeros([kXSize,kYSize,1,padding], dtype=tf.float32) #[5,5,1,4]
    KernelGroup = tf.concat([kernel, ZeroPad],3) #[5,5,1,36]
  else:
    KernelGroup = kernel
  print (KernelGroup.get_shape())
  KernelLine  = tf.split(KernelGroup, outy*outx, 3) #[5,5,1,1]

  k_map = tf.concat([tf.concat(KernelLine[0*outx:(0+1)*outx],0), tf.concat(KernelLine[1*outx:(1+1)*outx],0)], 1)
  for i in range(2,outy):
    k_map = tf.concat([k_map,tf.concat(KernelLine[i*outx:(i+1)*outx], 0)], 1)

  print (k_map.get_shape())
  k_map = tf.reshape(k_map, [1,kXSize*outx,kYSize*outy,1])
  return k_map

#input:   shape of a weights
#return:  initialized weight
def weight_variable(shape, std_value):
  initial = tf.truncated_normal(shape, dtype=tf.float32, stddev=std_value)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0., shape=shape)
  return tf.Variable(initial)


# input x and weight W 
#return the conv result
def conv2d(x,W):
  return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding = 'SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

def lrelu(x, leak=0.2, name="lrelu"):
  with tf.variable_scope(name):
    f1 = 0.5 * (1 + leak)
    f2 = 0.5 * (1 - leak)
    return f1 * x + f2 * abs(x)


def mlpconv(x, x_channel, conv_k_size, conv_k_num, mlp1_k_num, mlp2_k_num, has_2x2_pooling, weight_std_value):
  'construct mlp convolution layer'
  '(conv+relu)-->(MLP+relu)-->(MLE+relu)'
  'x          : input image'
  'x_channel  : x input number of channel'
  'conv_k_size: kernel x,y dimmension. x and y are same, so this is a scalar'
  'conv_k_num : number of kernel'
  'mlp1_k_num : mlp1 layer number of 1x1 kernel'
  'mlp2_k_num : mlp2 layer number of 1x1 kernel'
  'has_2x2_pooling: if do max pooling to the output 1 is yes 0 is no'

  #conv layer weights
  W_conv1 = weight_variable([conv_k_size, conv_k_size ,x_channel, conv_k_num], weight_std_value)
  b_conv1 = bias_variable([conv_k_num])

  W_mlp1  = weight_variable([1, 1 ,conv_k_num, mlp1_k_num], weight_std_value)
  b_mlp1  = bias_variable([mlp1_k_num])

  W_mlp2  = weight_variable([1, 1 ,mlp1_k_num, mlp2_k_num], weight_std_value)
  b_mlp2  = bias_variable([mlp2_k_num])  
  #network cascade
  h  = conv2d(x, W_conv1) + b_conv1
  h  = tf.nn.local_response_normalization(h, depth_radius=5, bias=0.01 )
  h  = lrelu(h)

  h  = conv2d(h, W_mlp1) + b_mlp1
  h  = tf.nn.local_response_normalization(h, depth_radius=5, bias=0.01 )
  h  = lrelu(h)

  h  = conv2d(h, W_mlp2) + b_mlp2
  h  = tf.nn.local_response_normalization(h, depth_radius=5, bias=0.01 )
  h  = lrelu(h)
  #h  = tf.nn.dropout(h, 0.5)

  if has_2x2_pooling==1:    
    h  = max_pool_2x2(h) 
  return h

def global_ave_pool(x, x_shape):
  'global average pooling the input x'
  'x       :input feature map [1,x,y,1]'
  'x_shape :should be [1,x,y,1]'
  return tf.nn.avg_pool(x, x_shape,[1,2,2,1], 'VALID')

def fc_relu_layer(x, in_length, out_length):
  w = weight_variable([in_length, out_length])
  b = bias_variable([out_length])

  y = tf.nn.relu(tf.matmul(x, w) + b)
  return y
def dropout_layer(x, rate):
  return tf.nn.dropout(x, rate)

def GetCIFARBatch(dataset, batch_size, pick_range):
  '''
  randomely get several image and label pairs from the dataset
  dataset:    dataset with format [image, label]
  batch_size: batch size
  return:     [image, label] 
  '''
  import random
  image = dataset[0]
  label = dataset[1]  

  batch_image=[]
  batch_label=[]
  for i in range(batch_size):
    idxf = random.random()
    idxf = idxf*pick_range
    idxint = int(idxf)
    #print(idxint)
    batch_image.append(image[idxint])
    batch_label.append(label[idxint])

  batch_image = np.array(batch_image)
  batch_label = np.array(batch_label)
  return (batch_image, batch_label)
  

def LoadCIFARName(file_name):
  '''
  load CIFAT-10 data set
  file_name:  the cifar-10 data file path
  return:  [train_image, train_label]
      train_image: image in shape [10000,3,32,32]
      train_label: label of each image in shape [10000,10]. with each item
                   [0,0,0,0,0,0,0,1,0,0]
  '''
  fo = open(file_name,'rb')
  datadict = cPickle.load(fo)
  fo.close()
  train_image = datadict["data"] 
  train_array = datadict['labels']
  train_image = train_image.reshape(10000, 3, 32, 32).transpose(0,2,3,1).astype("float")
  train_array = np.array(train_array)

  train_label=[]
  for i in range(len(train_array)):
    item=[0.,0.,0.,0.,0.,0.,0.,0.,0.,0.]
    item[train_array] = 1.
    train_label.append(item)

  train_label= np.array(train_label)
  return [train_image,train_label]
    
def ShowTenImage(data_set):
  labels = data_set[1]
  images = data_set[0]
  for i in range(40):
    plt.figure(1)
    plt.imshow(images[i,:,:,:])
    print(labels)
    plt.show()
  plt.show()

def image_normalize(img):
  '''
  normalize an image with shape [x,y,channel]
  '''
  mean      = np.mean(img)
  size = img.shape
  img = img-mean
  std = np.std(img)
  img = img/std
  return img


def image_set_normalize(img_set):
  '''
  normalize an image set with shape [N,x,y,channel]
  where N is number of image in the set
  channel is color channels
  '''
  size = img_set.shape
  for i in range(size[0]):
    img_set[i,:,:,:] = image_normalize(img_set[i,:,:,:])
  return img_set

def display_image_blocked(img):
  '''
  display a image in a new window program will be blocked until the wndows been close
  '''
  plt.imshow(img, cmap='gray')
  
  plt.show()
#['airplane',     0 
# 'automobile',   1
# 'bird',         2
# 'cat',          3
# 'deer',         4
# 'dog',          5
# 'frog',         6
# 'horse',        7  
# 'ship',         8
# 'truck']        9
#[6 9 9 4 1 1 2 7 8 3]


#open data set
#fo = open('cifar10data/data_batch_1','rb')
#datadict = cPickle.load(fo)
#fo.close()
#train_image = datadict["data"] 
#train_label = datadict['labels']
#train_image = train_image.reshape(10000, 3, 32, 32).transpose(0,2,3,1).astype("float")
#train_label = np.array(train_label)

#for i in range(10):
#  plt.figure(i)
#  plt.imshow(train_image[i,:,:,:])
#print(train_label[0:10])
#plt.show()
##image input 32x32




#command line arguments
# sys.argv[1]   if initialize all the weights from a file 
#   when "0"     do not initialize all the weights from a file
#   when "file"  initialize all the weights from file 
# sys.argv[2]   if write the final weights to a file
#   when "0"     do not 
#   when "file"  write to file


#check the number of input parameter
if len(sys.argv)!=3:
  print "too few arguement"
  quit()


train_batch_size = 20

train_set  = LoadCIFARName('cifar10data/data_batch_1')
train_set1 = LoadCIFARName('cifar10data/data_batch_2')
train_set2 = LoadCIFARName('cifar10data/data_batch_3')
train_set3 = LoadCIFARName('cifar10data/data_batch_4')
train_set4 = LoadCIFARName('cifar10data/data_batch_5')



#ShowTenImage(train_set4)


train_set[0]= np.concatenate((train_set[0], train_set1[0]), axis=0)
train_set[1]= np.concatenate((train_set[1], train_set1[1]), axis=0)

train_set[0]= np.concatenate((train_set[0], train_set2[0]), axis=0)
train_set[1]= np.concatenate((train_set[1], train_set2[1]), axis=0)

train_set[0]= np.concatenate((train_set[0], train_set3[0]), axis=0)
train_set[1]= np.concatenate((train_set[1], train_set3[1]), axis=0)

train_set[0]= np.concatenate((train_set[0], train_set4[0]), axis=0)
train_set[1]= np.concatenate((train_set[1], train_set4[1]), axis=0)
print(train_set[0].shape)
print(train_set[1].shape)

display_image_blocked(train_set[0][0,:,:,:])
image_set_normalize(train_set[0])



#train_set -= np.mean(train_set, axis=(1,2,3))
#train_set /= np.std (train_set, axis=(1,2,3))

#ShowTenImage(train_set)



#placeholders 
#x : input pixel arrays
#y_: output result
x  = tf.placeholder(tf.float32, shape=[None,32,32,3])
y_ = tf.placeholder(tf.float32, shape=[None, 10])

x_image = tf.reshape(x, [-1,32,32,3])

#Network structure
network1 = mlpconv(x_image,  3 ,  5, 192, 160, 96,  1, 0.01)
network2 = mlpconv(network1, 96,  5, 192, 192, 192, 1, 0.01)
network3 = mlpconv(network2, 192, 3, 192, 192, 10,  0, 0.1)
ave_vec  = global_ave_pool(network3, [1,8,8,1])
h        = tf.reshape(ave_vec, [-1, 10])
y_conv   = h








#h = fc_relu_layer(h, 128, 256)
#h = dropout_layer(h, 0.3)
#h = fc_relu_layer(h, 256, 10)

ave = tf.Print(ave_vec, [ave_vec])
#dropout
keep_prob = tf.placeholder(tf.float32)
#h_fc1_drop = tf.nn.dropout(h_pool2_flat, keep_prob)

#readout layer
#W_fc2 = weight_variable([10, 10])
#b_fc2 = bias_variable([10])



#y_conv = tf.matmul(h_pool2_flat, W_fc2) + b_fc2
#y_conv = tf.nn.softmax(y_conv, -1)




#training and model
cross_entropy      = tf.reduce_mean(
                     tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
train_step         = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
#train_step         = tf.train.GradientDescentOptimizer(0.0001).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy           = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

saver = tf.train.Saver()
init = tf.initialize_all_variables()
sess = tf.Session()

#initialize from a file or not
if sys.argv[1]=="0":
  sess.run(init)
  print "Random initialize all weights bias"
else:
  saver.restore(sess, sys.argv[1])
  print "Restore weights and bias from file:",sys.argv[1]

#tb
#concatenate kernel map for conv layer1  
#KernelMap1=concatenate_kernel_map(W_conv1, 5, 5, 6, 6, 4)#KernelMap0=concatenate_kernel_map(W_conv1)
#KernelMap1 = tf.reshape(KernelMap1, [1,30,30,1])


#output_fm = tf.split(output_fm,train_batch_size,0)
#output_fm = tf.concat([output_fm[0:10]],3)
#for i in range(10):
#  feature_map = tf.reshape(output_fm, [4,4,1,10])
#  global_ave_fm = concatenate_kernel_map(feature_map, 4, 4, 1, 10, 0)
#  fm_out        = tf.summary.image("before global ave pooling 0", global_ave_fm, max_outputs=1)

#print(global_ave_fm.get_shape())
#global_ave_fm = tf.reshape(global_ave_fm, [1,35,14,1])
#fm_out        = tf.summary.image("before global ave pooling", global_ave_fm, max_outputs=1)
#concatenate kernel map for conv layer2 
#KernelMap2Group  = tf.split(W_conv2, 32, 2)
#KernelMap2Group2  = KernelMap2Group[0]#tf.split(KernelMap2Group[0], 2, 3)
#print ('dimision kernelgroup2', KernelMap2Group2.get_shape())
#KernelMap2=concatenate_kernel_map(KernelMap2Group2, 5, 5, 8, 8, 0)#KernelMap20=concatenate_kernel_map(KernelMap2Group2[0])
#KernelMap2 = tf.reshape(KernelMap2, [1,40,40,1])

#conv1_k = tf.summary.image("conv1 kernels", KernelMap1, max_outputs=1)
#conv2_k = tf.summary.image("conv2 kernels", KernelMap2, max_outputs=1)
accuracy_chart = tf.summary.scalar("accuracy", accuracy)
#weights_conv1 = tf.summary.histogram("global average pooling", ave_vec)



merged_summary_op = tf.summary.merge_all()
summary_writer = tf.summary.FileWriter('/home/yuf/dev/deep_learning/prj/CIFAR/log', sess.graph)

label_name = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']
label_name = np.arange(len(label_name))
#training loop
for i in range(300000):
  batch = GetCIFARBatch(train_set, train_batch_size, 50000)
  if i%50 == 0:

  
    train_accuracy = sess.run(accuracy, feed_dict={x:batch[0], y_: batch[1], keep_prob: 1.0})
    summary_str    = sess.run(merged_summary_op, feed_dict={x:batch[0], y_: batch[1], keep_prob: 1.0})
    summary_writer.add_summary(summary_str, i)
    print("step %d, training accuracy %g"%(i, train_accuracy))
    print(batch[1][0,:])
    ave_vec_num = sess.run(ave_vec, feed_dict={x:batch[0], y_: batch[1], keep_prob: 1.0})
    print(ave_vec_num[0,0,0,:])
    #plt.bar(label_name, ave_vec_num[0,0,0,:], alpha=0.5)
    #plt.ion()
    #plt.show()
    #plt.pause(0.05)

  if i%1000 == 0:
    if sys.argv[2]=="0":
      sess.run(init)
      print "Discard all weights & bias"
    else:
      saver.save(sess, sys.argv[2])
      print "Save weights and bias to file:",sys.argv[2]
  sess.run(train_step, feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
  


test_set   = LoadCIFARName('cifar10data/test_batch')
test_set[0]   = image_set_normalize(test_set[0])

test_batch      = GetCIFARBatch(test_set, 500, 10000)
print("test accuracy %g"%sess.run(accuracy, feed_dict={
    x: test_batch[0], y_:test_batch[1], keep_prob: 1.0}))
test_batch      = GetCIFARBatch(test_set, 500, 10000)
print("test accuracy %g"%sess.run(accuracy, feed_dict={
    x: test_batch[0], y_:test_batch[1], keep_prob: 1.0}))

test_batch      = GetCIFARBatch(test_set, 500, 10000)
print("test accuracy %g"%sess.run(accuracy, feed_dict={
    x: test_batch[0], y_:test_batch[1], keep_prob: 1.0}))

test_batch      = GetCIFARBatch(test_set, 500, 10000)
print("test accuracy %g"%sess.run(accuracy, feed_dict={
    x: test_batch[0], y_:test_batch[1], keep_prob: 1.0}))

test_batch      = GetCIFARBatch(test_set, 500, 10000)
print("test accuracy %g"%sess.run(accuracy, feed_dict={
    x: test_batch[0], y_:test_batch[1], keep_prob: 1.0}))


if sys.argv[2]=="0":
  sess.run(init)
  print "Discard all weights & bias"
else:
  saver.save(sess, sys.argv[2])
  print "Save weights and bias to file:",sys.argv[2]


sess.close()