import tensorflow as tf

with tf.name_scope('input'):
    xs = tf.placeholder(tf.float32,[None,1])
    ys = tf.placeholder(tf.float32,[None,1])

def add_layer(inputs,in_size,out_size,activation_function=None):
    with tf.name_scope("layer"):
        with tf.name_scope("weights"):
            Weights = tf.Variable(tf.random_normal([out_size,in_size]),name='W')
        with tf.name_scope("biases"):
            biases = tf.Variable(tf.zeros([1,out_size])+0.1,name="b")
        with tf.name_scope("affine"):
            affine = tf.matmul(Weights,inputs)+biases
        with tf.name_scope("activation"):
            if activation_function is None:
                outputs = affine
            else:
                outputs = activation_function(affine)
        return outputs

l1 = add_layer(xs,1,10,tf.nn.relu)
hypothesis = add_layer(l1,10,1,tf.nn.softmax)

with tf.name_scope("loss"):
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys-hypothesis),axis=1))

with tf.name_scope("train"):
    train_step = tf.train.GradientDescentOptimizer(0.2)
sess = tf.Session()
writer = tf.summary.FileWriter("logs/",sess.graph)
init = tf.global_variables_initializer()
sess.run(init)
