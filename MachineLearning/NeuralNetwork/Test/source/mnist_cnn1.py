import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data",one_hot=True)
x=tf.placeholder(tf.float32,[None,784])
y_ = tf.placeholder(tf.float32,[None,10])
keep_prob = tf.placeholder(tf.float32)
x_image = tf.reshape(x,[-1,28,28,1])

W_conv1 = tf.Variable(tf.truncated_normal([5,5,1,16]))
b_conv1 = tf.Variable(tf.truncated_normal([16]))
h_conv1 = tf.nn.relu(tf.nn.conv2d(x_image,W_conv1,strides=[1,1,1,1],padding="SAME")+b_conv1)
h_pooling = tf.nn.max_pool(h_conv1,ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME")

W_conv2 = tf.Variable(tf.truncated_normal([5,5,16,32]))
b_conv2 = tf.Variable(tf.truncated_normal([32]))
h_conv2 = tf.nn.relu(tf.nn.conv2d(h_pooling,W_conv2,strides=[1,1,1,1],padding="SAME")+b_conv2)
h_pooling = tf.nn.max_pool(h_conv2,ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME")
h_pooling_flat = tf.reshape(h_pooling,[-1,7*7*32])

W_fc1 = tf.Variable(tf.truncated_normal([7*7*32,1024]))
b_fc1 = tf.Variable(tf.truncated_normal([1024]))
h_fc1 = tf.matmul(h_pooling_flat,W_fc1)+b_fc1
h_fc1_dropout = tf.nn.dropout(h_fc1,keep_prob=keep_prob)

W_fc2 = tf.Variable(tf.truncated_normal([1024,10]))
b_fc2 = tf.Variable(tf.truncated_normal([10]))
h_fc2 = tf.matmul(h_fc1_dropout,W_fc2)+b_fc2

cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=h_fc2,labels=y_)
train_step = tf.train.AdamOptimizer(1e-3).minimize(cross_entropy)
accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(h_fc2,axis=1),tf.argmax(y_,axis=1)),tf.float32))

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for i in range(5000):
    batch = mnist.train.next_batch(100)
    if i%10 == 0:
        print("step%d accuracy%g" %(i,sess.run(accuracy,feed_dict={x:batch[0],y_:batch[1],keep_prob:1})))

    sess.run(train_step, feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.4})

print("training set accuracy %g" %sess.run(accuracy,feed_dict={x:mnist.test.images,y_:mnist.test.labels,keep_prob:1}))