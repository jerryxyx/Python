from __future__ import print_function, division
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

num_epochs = 10
total_series_length = 50000
truncated_backprop_length = 15
state_size = 4
num_classes = 2
echo_step = 3
batch_size = 5
num_batches = total_series_length//batch_size//truncated_backprop_length

def generateData():
    x = np.array(np.random.choice(2, total_series_length, p=[0.5, 0.5]))
    y = np.roll(x, echo_step)
    y[0:echo_step] = 0

    x = x.reshape((batch_size, -1))  # The first index changing slowest, subseries as rows
    y = y.reshape((batch_size, -1))

    return (x, y)

batchX_placeholder = tf.placeholder(tf.float32, [batch_size, truncated_backprop_length])
batchY_placeholder = tf.placeholder(tf.int32, [batch_size, truncated_backprop_length])

init_state = tf.placeholder(tf.float32, [batch_size, state_size])
# ##########################################################################################
# Wa = tf.Variable(np.random.randn(1,state_size),dtype=tf.float32)
# Wb = tf.Variable(np.random.randn(state_size,state_size),dtype=tf.float32)
# ##########################################################################################
W = tf.Variable(np.random.rand(state_size+1, state_size), dtype=tf.float32)
b = tf.Variable(np.zeros((1,state_size)), dtype=tf.float32)

W2 = tf.Variable(np.random.rand(state_size, num_classes),dtype=tf.float32)
b2 = tf.Variable(np.zeros((1,num_classes)), dtype=tf.float32)

# ##########################################################################################
offset_out = tf.Variable(np.random.randn(1),dtype=tf.float32)
scale_out = tf.Variable(0.25,dtype=tf.float32)
variance_epsilon_out = tf.constant(0.00001,dtype=tf.float32)
# offset_input = tf.Variable(np.random.randn(1),dtype=tf.float32)
# scale_input = tf.Variable(1,dtype=tf.float32)
# variance_epsilon_input = tf.constant(0.00001,dtype=tf.float32)
# offset_state = tf.Variable(np.random.randn(1),dtype=tf.float32)
# scale_state = tf.Variable(1,dtype=tf.float32)
# variance_epsilon_state = tf.constant(0.00001,dtype=tf.float32)
# ##########################################################################################

# Unpack columns
inputs_series = tf.unstack(batchX_placeholder, axis=1)
labels_series = tf.unstack(batchY_placeholder, axis=1)

# Forward pass
current_state = init_state
states_series = []
for current_input in inputs_series:
    current_input = tf.reshape(current_input, [batch_size, 1])
    input_and_state_concatenated = tf.concat(values=[current_input, current_state], axis=1)  # Increasing number of columns

    ##########################################################################################
    # # batch norm start
    # (mean_input,var_input) = tf.nn.moments(current_input,axes=[0])
    # (mean_state, var_state) = tf.nn.moments(current_state, axes=[0])
    # current_input_norm = tf.nn.batch_normalization(current_input,mean_input,var_input,
    #     offset=offset_input, scale=scale_input, variance_epsilon=variance_epsilon_input)
    # current_state_norm = tf.nn.batch_normalization(current_state, mean_state, var_state,
    #     offset=offset_state, scale=scale_state, variance_epsilon=variance_epsilon_state)
    # next_state = tf.tanh(tf.matmul(current_input_norm, Wa) + tf.matmul(current_state_norm, Wb))  # Split concatenation
    # # batch norm end
    ##########################################################################################
    next_state = tf.tanh(tf.matmul(input_and_state_concatenated, W) + b)  # Broadcasted addition
    # next_state = tf.tanh(tf.matmul(Wa_with_memory, current_input) + b_with_memory)  # Split concatenation
    states_series.append(next_state)
    current_state = next_state

logits_series = [tf.matmul(state, W2) + b2 for state in states_series] #Broadcasted addition

##########################################################################################
# batch norm of activated tensor start
logits_series_norm = []
for logits in logits_series:
    (mean,var) = tf.nn.moments(logits,axes=[0])
    logits_series_norm.append(tf.nn.batch_normalization(logits, mean, var,
        offset=offset_out,scale=scale_out,variance_epsilon=variance_epsilon_out))
# batch norm end
##########################################################################################

# ##########################################################################################
# # batch norm of Z(input tensor) start
# states_series_norm = []
# for states in states_series:
#     (mean,var) = tf.nn.moments(states,axes=[0])
#     states_series_norm.append(tf.nn.batch_normalization(states, mean, var,
#         offset=offset_out,scale=scale_out,variance_epsilon=variance_epsilon_out))
# logits_series_norm = [tf.matmul(states, W2) + b2 for states in states_series_norm]
# # batch norm end
# ##########################################################################################

# predictions_series = [tf.nn.softmax(logits) for logits in logits_series]
predictions_series = [tf.nn.softmax(logits) for logits in logits_series_norm]

losses = [tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels) for logits, labels in zip(logits_series,labels_series)]
total_loss = tf.reduce_mean(losses)

train_step = tf.train.AdagradOptimizer(0.3).minimize(total_loss)

def plot(loss_list, predictions_series, batchX, batchY):
    plt.subplot(2, 3, 1)
    plt.cla()
    plt.plot(loss_list)

    for batch_series_idx in range(batch_size):
        one_hot_output_series = np.array(predictions_series)[:, batch_series_idx, :]
        single_output_series = np.array([(1 if out[0] < 0.5 else 0) for out in one_hot_output_series])

        plt.subplot(2, 3, batch_series_idx + 2)
        plt.cla()
        plt.axis([0, truncated_backprop_length, 0, 2])
        left_offset = range(truncated_backprop_length)
        plt.bar(left_offset, batchX[batch_series_idx, :], width=1, color="blue")
        plt.bar(left_offset, batchY[batch_series_idx, :] * 0.5, width=1, color="red")
        plt.bar(left_offset, single_output_series * 0.3, width=1, color="green")

    plt.draw()
    plt.pause(0.0001)


with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    plt.ion()
    plt.figure()
    plt.show()
    loss_list = []

    for epoch_idx in range(num_epochs):
        x,y = generateData()
        _current_state = np.zeros((batch_size, state_size))

        print("New data, epoch", epoch_idx)

        for batch_idx in range(num_batches):
            start_idx = batch_idx * truncated_backprop_length
            end_idx = start_idx + truncated_backprop_length

            batchX = x[:,start_idx:end_idx]
            batchY = y[:,start_idx:end_idx]

            _total_loss, _train_step, _current_state, _predictions_series= sess.run(
                [total_loss, train_step, current_state, predictions_series],
                feed_dict={
                    batchX_placeholder:batchX,
                    batchY_placeholder:batchY,
                    init_state:_current_state
                })

            loss_list.append(_total_loss)

            if batch_idx%100 == 0:
                print("Step",batch_idx, "Loss", _total_loss)
                plot(loss_list, _predictions_series, batchX, batchY)

plt.ioff()
plt.show()