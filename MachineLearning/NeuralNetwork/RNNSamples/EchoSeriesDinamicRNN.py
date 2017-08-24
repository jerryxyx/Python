import numpy as np
import tensorflow as tf
# %matplotlib inline
import matplotlib.pyplot as plt
# Global config variables
num_steps = 5 # number of truncated backprop steps ('n' in the discussion above)
batch_size = 200
num_classes = 2
state_size = 4
learning_rate = 0.1
num_epochs = 10

def gen_data(size=1000000):
    X = np.array(np.random.choice(2, size=(size,)))
    Y = []
    for i in range(size):
        threshold = 0.5
        if X[i-3] == 1:
            threshold += 0.5
        if X[i-8] == 1:
            threshold -= 0.25
        if np.random.rand() > threshold:
            Y.append(0)
        else:
            Y.append(1)
    return X, np.array(Y)

# adapted from https://github.com/tensorflow/tensorflow/blob/master/tensorflow/models/rnn/ptb/reader.py
def gen_batch(raw_data, batch_size, num_steps):
    raw_x, raw_y = raw_data
    data_length = len(raw_x)

    # partition raw data into batches and stack them vertically in a data matrix
    batch_partition_length = data_length // batch_size
    data_x = np.zeros([batch_size, batch_partition_length], dtype=np.int32)
    data_y = np.zeros([batch_size, batch_partition_length], dtype=np.int32)
    for i in range(batch_size):
        data_x[i] = raw_x[batch_partition_length * i:batch_partition_length * (i + 1)]
        data_y[i] = raw_y[batch_partition_length * i:batch_partition_length * (i + 1)]
    # further divide batch partitions into num_steps for truncated backprop
    epoch_size = batch_partition_length // num_steps

    for i in range(epoch_size):
        x = data_x[:, i * num_steps:(i + 1) * num_steps]
        y = data_y[:, i * num_steps:(i + 1) * num_steps]
        yield (x, y)

def gen_epochs(n, num_steps):
    for i in range(n):
        yield gen_batch(gen_data(), batch_size, num_steps)


"""
Placeholders
"""

x = tf.placeholder(tf.int32, [batch_size, num_steps], name='input_placeholder')
y = tf.placeholder(tf.int32, [batch_size, num_steps], name='labels_placeholder')
init_state = tf.zeros([batch_size, state_size], name='initial_state')

"""
Inputs
"""

"""
static
"""
# x_one_hot = tf.one_hot(x, num_classes)
# rnn_inputs = tf.unstack(x_one_hot, axis=1)

rnn_inputs = tf.one_hot(x, num_classes, name='rnn_inputs')

"""
RNN
"""

cell = tf.contrib.rnn.BasicRNNCell(state_size)
# rnn_outputs, final_state = tf.contrib.rnn.static_rnn(cell, rnn_inputs, initial_state=init_state)
rnn_outputs, final_state = tf.nn.dynamic_rnn(cell, rnn_inputs, initial_state=init_state)
"""
Predictions, loss, training step
"""

with tf.variable_scope('softmax'):
    W = tf.get_variable('W', [state_size, num_classes])
    b = tf.get_variable('b', [num_classes], initializer=tf.zeros_initializer())
"""
static
"""
# logits = [tf.matmul(rnn_output, W) + b for rnn_output in rnn_outputs]
# predictions = [tf.nn.softmax(logit) for logit in logits]
logits = tf.reshape(
            tf.matmul(tf.reshape(rnn_outputs, [-1, state_size]), W) + b,
            [batch_size, num_steps, num_classes])

predictions = tf.nn.softmax(logits)
"""
static
"""
# y_as_list = tf.unstack(y, num=num_steps, axis=1)
#
# losses = [tf.nn.sparse_softmax_cross_entropy_with_logits(labels=label, logits=logit) for \
#           logit, label in zip(logits, y_as_list)]
losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)

total_loss = tf.reduce_mean(losses)
train_step = tf.train.AdagradOptimizer(learning_rate).minimize(total_loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    training_losses = []
    for idx, epoch in enumerate(gen_epochs(num_epochs, num_steps)):
        training_loss = 0
        training_state = np.zeros((batch_size, state_size))

        print("\nEPOCH", idx)
        for step, (X, Y) in enumerate(epoch):
            tr_losses, training_loss_, training_state, _ = \
                sess.run([losses,
                          total_loss,
                          final_state,
                          train_step],
                         feed_dict={x: X, y: Y, init_state: training_state})
            training_loss += training_loss_
            if step % 100 == 0 and step > 0:
                print("Average loss at step", step,
                      "for last 250 steps:", training_loss / 100)
                training_losses.append(training_loss / 100)
                training_loss = 0