
import tensorflow as tf;
from tensorflow.examples.tutorials.mnist import input_data
 
class DNNMnist:
    def __init__(self):
        self.batch_size = 100;
        self.epoches = 10000;
        self.n_hiddenlayers = 2;
        self.n_classes =10;
        self.input_features = 784;
        self.regularizer_rate = 0.00001
        self.mnist = input_data.read_data_sets("MNIST_data/", one_hot = True)
    def Train(self):
        print ("training....")
        weights={
            'W1':tf.Variable(tf.random_normal(shape=[self.input_features,512],stddev=0.01)),
            "W2":tf.Variable(tf.random_normal(shape=[512,256],stddev=0.01)),
            "Output":tf.Variable(tf.random_normal(shape=[256,self.n_classes],stddev=0.01)),
            }
        
        bias = {
            'b1':tf.Variable(tf.zeros([512]),dtype=tf.float32),
            'b2':tf.Variable(tf.zeros([256]),dtype=tf.float32),
            'Output':tf.Variable(tf.zeros([10]),dtype=tf.float32)
            }
        
        x = tf.placeholder(tf.float32,shape=[None,self.input_features])
        y = tf.placeholder(tf.float32,shape=[None,self.n_classes])
        
        output1 = tf.nn.relu(tf.matmul(x,weights['W1'])+bias['b1']);
        output2 = tf.nn.relu(tf.matmul(output1,weights['W2'])+bias['b2']);
        output  = tf.matmul(output2,weights['Output']) + bias['Output'];
        
        y_ = tf.nn.softmax(output);
        
        #loss = tf.reduce_mean(tf.negative(tf.log(y_))*y);
        regularizer = tf.contrib.layers.l2_regularizer(self.regularizer_rate);
        with tf.name_scope("loss"):
            loss = tf.reduce_sum(-tf.log(y_) * y) + regularizer(weights['W1']) + regularizer(weights['W2']);
        
        optimizer_op = tf.train.GradientDescentOptimizer(0.001).minimize(loss);
        
        init_op = tf.global_variables_initializer();
        
        with tf.name_scope('accuracy'):
            accuracy_op = tf.reduce_mean(tf.cast(tf.equal(tf.arg_max(y_,1),tf.arg_max(y,1)),dtype=tf.float32));
            
        loss_summary = tf.summary.scalar("loss", loss)
        accuracy_summary = tf.summary.scalar("accuracy",accuracy_op)
        
        with tf.Session() as sess:
            sess.run(init_op);
            summary_op = tf.summary.merge_all()
            test_feed = {x:self.mnist.test.images,y:self.mnist.test.labels};
            #print ("accuracy:",sess.run(accuracy_op,feed_dict=test_feed));
            summary_writer = tf.summary.FileWriter("./log/",sess.graph);
            for i in range(self.epoches):
                batch_x,batch_y = self.mnist.train.next_batch(self.batch_size)
                train_feed = {x:batch_x,y:batch_y}
                
                sess.run(optimizer_op,feed_dict = train_feed);
                summary_output=sess.run(summary_op,feed_dict=train_feed);
                summary_writer.add_summary(summary_output, i);
                if (i % 100 == 0):
                    print ("loss:",sess.run(loss,feed_dict=train_feed))
            
            test_feed = {x:self.mnist.test.images,y:self.mnist.test.labels};
            print ("accuracy:",sess.run(accuracy_op,feed_dict=test_feed));
            summary_writer.close()
if __name__ == "__main__":
    dnnMnist = DNNMnist();
    dnnMnist.Train();
