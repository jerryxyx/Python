import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

n_fc1_neurons = 1024

mnist = input_data.read_data_sets("MNIST_data",one_hot=True)
x = tf.placeholder(tf.float32,[None,784])
y_ = tf.placeholder(tf.float32,[None,10])
x_image = tf.reshape(x,[-1,28,28,1])

W_conv1 = tf.Variable(tf.truncated_normal([5,5,1,32]))
b_conv1 = tf.Variable(tf.truncated_normal([32]))
h_conv1 = tf.nn.relu(tf.nn.conv2d(input=x_image,filter=W_conv1,strides=[1,1,1,1],padding="SAME")+b_conv1)
h_maxpool1 = tf.nn.max_pool(h_conv1,ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME")

W_conv2 = tf.Variable(tf.truncated_normal([5,5,32,64]))
b_conv2 = tf.Variable(tf.truncated_normal([64]))
h_conv2 = tf.nn.relu(tf.nn.conv2d(h_maxpool1,filter=W_conv2,strides=[1,1,1,1],padding="SAME")+b_conv2)
h_maxpool2 = tf.nn.max_pool(h_conv2,ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME")

h_maxpool2_flat = tf.reshape(h_maxpool2,[-1,7*7*64])
W_fc1 = tf.Variable(tf.truncated_normal([7*7*64,n_fc1_neurons]))
b_fc1 = tf.Variable(tf.truncated_normal([n_fc1_neurons]))
h_fc1 = tf.nn.relu(tf.matmul(h_maxpool2_flat,W_fc1)+b_fc1)


#dropout
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1,keep_prob=keep_prob)


W_fc2 = tf.Variable(tf.truncated_normal([n_fc1_neurons,10]))
b_fc2 = tf.Variable(tf.truncated_normal([10]))
#h_fc2 = tf.matmul(h_fc1,W_fc2)+b_fc2
h_fc2 = tf.matmul(h_fc1_drop,W_fc2)+b_fc2

cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_,logits=h_fc2)
accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y_,axis=1),tf.argmax(h_fc2,axis=1)),tf.float32))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
for i in range(3000):
    batch = mnist.train.next_batch(25)
    sess.run(train_step,feed_dict={x:batch[0],y_:batch[1],keep_prob:0.5})
    if i%10==0:
        train_accuracy = sess.run(accuracy,feed_dict={x:batch[0],y_:batch[1],keep_prob:1})
        print("step %d, training accuracy %g"%(i,train_accuracy))

print("the accuracy %g"%sess.run(accuracy,feed_dict={x:mnist.test.images,y_:mnist.test.labels,keep_prob:1}))
