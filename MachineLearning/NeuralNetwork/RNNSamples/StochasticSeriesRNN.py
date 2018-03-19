import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.python.framework import ops

ops.reset_default_graph()
num_epochs = 100
total_series_length = 50000
truncated_backprop_length = 32
state_size = 4
# cell_state_size_array = [4,4,4]
# hidden_state_size_array = [4,4,4]
# state_size_array = [4,8,4]
num_classes = 2
num_layers = 3
echo_step = 4
batch_size = 5
num_batches = total_series_length // batch_size // truncated_backprop_length
learning_rate = 0.3


def generateData(size=total_series_length, echo_step=echo_step):
    x = np.array(np.random.choice(2, size, p=[0.5, 0.5]))
    y = np.roll(x, echo_step)
    y[0:echo_step] = 0
    # x = x.reshape((batch_size, -1))  # The first index changing slowest, subseries as rows
    # y = y.reshape((batch_size, -1))
    return (x, y)

def print_expected_cross_entropy():
    print("Expected cross entropy loss if the model:")
    print("- learns neither dependency:", -(0.625 * np.log(0.625) +
                                            0.375 * np.log(0.375)))
    # Learns first dependency only ==> 0.51916669970720941
    print("- learns first dependency:  ",
          -0.5 * (0.875 * np.log(0.875) + 0.125 * np.log(0.125))
          - 0.5 * (0.625 * np.log(0.625) + 0.375 * np.log(0.375)))
    print("- learns both dependencies: ", -0.50 * (0.75 * np.log(0.75) + 0.25 * np.log(0.25))
          - 0.25 * (2 * 0.50 * np.log(0.50)) - 0.25 * (0))

print_expected_cross_entropy()
def gen_data(size=total_series_length):
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
        # raw_data = generateData()
        raw_data = gen_data(total_series_length)
        yield gen_batch(raw_data, batch_size, num_steps)

batchX_placeholder = tf.placeholder(tf.float32, [batch_size, truncated_backprop_length], name='input_placeholder')
batchY_placeholder = tf.placeholder(tf.int32, [batch_size, truncated_backprop_length], name='labels_placeholder')
# states_placeholder = tf.placeholder(tf.float32,[num_layers, 2, batch_size, state_size], name='states_placeholder')
rnn_state_placeholder = tf.placeholder(tf.float32,[batch_size, state_size], name='rnn_states_placeholder')
# rnn_inputs = tf.one_hot(batchX_placeholder, num_classes, name='rnn_inputs')
# labels = batchY_placeholder
# Unpack columns
inputs_series = tf.split(batchX_placeholder,axis=1,num_or_size_splits=truncated_backprop_length)
labels_series = tf.unstack(batchY_placeholder, axis=1)
# def gen_multilayers_state_tuple(rnn_state):
#     n_layers=rnn_state.shape[0]
#     rnn_state_tuple = tuple([tf.nn.rnn_cell.LSTMStateTuple(rnn_state[layer_idx,0],rnn_state[layer_idx,1])
#                              for layer_idx in range(n_layers)])
#     return rnn_state_tuple

# def gen_multilayers_rnn_state_tuple(rnn_state):
#     n_layers=rnn_state.shape[0]
#     rnn_state_tuple = tuple([(rnn_state[layer_idx,0])
#                              for layer_idx in range(n_layers)])
#     return rnn_state_tuple


# def gen_multilayers_LSTM_cell(state_size,num_layers):
#     cell = tf.nn.rnn_cell.MultiRNNCell(
#         [tf.nn.rnn_cell.LSTMCell(state_size,state_is_tuple=True)
#          for layer_idx in range(num_layers)],
#         state_is_tuple=True
#     )
#     return cell

# def gen_multilayers_RNN_cell(state_size,num_layers):
#     cell = tf.nn.rnn_cell.MultiRNNCell(
#         [tf.nn.rnn_cell.RNNCell(state_size)
#          for layer_idx in range(num_layers)]
#     )
#     return cell

# cell = gen_multilayers_RNN_cell(state_size,num_layers)
# init_state = gen_multilayers_rnn_state_tuple(rnn_states_placeholder)
# rnn_outputs, final_state = tf.nn.dynamic_rnn(cell, rnn_inputs, initial_state=init_state)

cell = tf.contrib.rnn.BasicRNNCell(state_size)
rnn_outputs, final_state = tf.contrib.rnn.static_rnn(cell, inputs_series, rnn_state_placeholder)

with tf.variable_scope('softmax'):
    W = tf.get_variable('W', [state_size, num_classes])
    b = tf.get_variable('b', [num_classes], initializer=tf.zeros_initializer())

with tf.variable_scope('output'):
    # logits = tf.reshape(tf.matmul(tf.reshape(rnn_outputs, [-1, state_size]), W) + b,
    #                [batch_size, truncated_backprop_length, num_classes],name='logits')
    # predictions = tf.nn.softmax(logits,name="predictions")
    # losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)
    # total_loss = tf.reduce_mean(losses)
    # train_step = tf.train.AdagradOptimizer(learning_rate).minimize(total_loss)
    logits_series = [tf.matmul(output, W) + b for output in rnn_outputs]  # Broadcasted addition
    predictions_series = [tf.nn.softmax(logits) for logits in logits_series]
    losses = [tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels) for logits, labels in
              zip(logits_series, labels_series)]

total_loss = tf.reduce_mean(losses)
train_step = tf.train.AdagradOptimizer(learning_rate).minimize(total_loss)

def plot(loss_list):
    # plt.subplot(2, 3, 1)
    plt.cla()
    plt.plot(loss_list)
    plt.draw()
    plt.pause(0.0001)

# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     plt.ion()
#     plt.figure()
#     plt.show()
#     loss_list = []
#
#     for epoch_idx, epoch in enumerate(gen_epochs(num_epochs, num_steps=truncated_backprop_length)):
#         _current_state = np.zeros((batch_size, state_size))
#         print("New data, epoch", epoch_idx)
#         for batch_idx, batch in enumerate(epoch):
#             start_idx = batch_idx * truncated_backprop_length
#             end_idx = start_idx + truncated_backprop_length
#
#             batchX = batch[0]
#             batchY = batch[1]
#
#             _total_loss, _train_step, _current_state, _predictions_series = sess.run(
#                 [total_loss, train_step, final_state, predictions_series],
#                 feed_dict={
#                     batchX_placeholder:batchX,
#                     batchY_placeholder:batchY,
#                     rnn_state_placeholder:_current_state
#                 })
#
#             loss_list.append(_total_loss)
#
#             if batch_idx%100 == 0:
#                 print("Step",batch_idx, "Loss", _total_loss)
#                 plot(loss_list, _predictions_series, batchX, batchY)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    plt.ion()
    plt.figure()
    plt.show()
    loss_list = []
    for epoch_idx, epoch in enumerate(gen_epochs(num_epochs,num_steps=truncated_backprop_length)):
        print("Epoch",epoch_idx)
        # _current_state = sess.run(cell.zero_state(batch_size=batch_size,dtype=tf.float32))
        _current_state = np.zeros((batch_size, state_size))
        for batch_idx, batch in enumerate(epoch):
            _total_loss, _train_step, _current_state = sess.run(
                [total_loss, train_step, final_state],
                feed_dict={
                    batchX_placeholder: batch[0],
                    batchY_placeholder: batch[1],
                    rnn_state_placeholder: _current_state
                })
            loss_list.append(_total_loss)

            if batch_idx % 100 == 0:
                print("Step", batch_idx, "Loss", _total_loss)
                plot(loss_list)


plt.ioff()
plt.show()