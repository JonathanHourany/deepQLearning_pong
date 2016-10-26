from deepQAlg import DeepQLearner
import tensorflow as tf
import numpy as np
import gym


def init_weight_matrix(shape):
    initial_values = tf.truncated_normal(shape, stddev=0.01)
    return tf.Variable(initial_values)


def init_bias_matrix(shape):
    initial_values = tf.constant(0.01, shape=shape)
    return tf.Variable(initial_values)


def conv2d(x, W, stride):
    return tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def createNetwork(actions):

    # NOTE: About max_pooling -- There are 8 directions in which one can translate
    #   the input image by a single pixel. If max-pooling is done over a 2x2 region,
    #   3 out of these 8 possible configurations will produce exactly the same output at
    #   the convolutional layer. This causes translation invareance which is likely
    #   NOT what we want here!

    # Input Layer
    state = tf.placeholder(tf.float32, [None, 80, 80, 4], name='state_pl')

    # Hidden Layers
    W_conv1 = init_weight_matrix([8, 8, 4, 32])
    b_conv1 = init_bias_matrix([32])
    h_conv1 = tf.nn.relu(conv2d(state, W_conv1, 4) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)
    # h_pool1 shape = (?, 10, 10, 32)

    W_conv2 = init_weight_matrix([4, 4, 32, 64])
    b_conv2 = init_bias_matrix([64])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2, 2) + b_conv2)
    # h_conv2 shape = (?, 5, 5, 64)

    #h_conv3_flat = tf.reshape(h_conv3, [-1, 1600])
    h_conv2_flat = tf.reshape(h_conv2, [-1, 1600])

    W_fc1 = init_weight_matrix([1600, 400])
    b_fc1 = init_bias_matrix([400])
    h_fc1 = tf.nn.relu(tf.matmul(h_conv2_flat, W_fc1) + b_fc1)
    # h_fc1 shape = (?, 400)

    # Output Layer
    W_fc2 = init_weight_matrix([400, actions])
    b_fc2 = init_bias_matrix([actions])
    output = tf.matmul(h_fc1, W_fc2) + b_fc2
    # Final output is a (?, 6) array

    return state, output, h_fc1

def lets_play():
    # Initialize Enviroment
    sess = tf.InteractiveSession()
    env = gym.make('Pong-v0')
    state, nn_readout, h_fc1 = createNetwork(env.action_space.n)
    dqn = DeepQLearner(game_env=env, neural_network=nn_readout, skip_n_frames=3)
    dqn.train_network(sess, state)

def main():
    lets_play()

if __name__ == "__main__":
    main()
