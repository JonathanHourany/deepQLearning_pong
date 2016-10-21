import matplotlib.pyplot as plt # Saving images
from collections import deque
import tensorflow as tf
import numpy as np
import random
import sys
import gym
import cv2 # For image preprocessing. Will be replacing soon

## Hyperparameters
learning_rate = 1e-2
MEMORY_LENGTH = 1000000
NUM_EPISODES = 5000000
INIT_EPSILON = .1
BATCH_SIZE = 32
NUM_FRAMES_SKIP = 4
decay_rate = 0.99  # decay factor for RMSProp leaky sum of grad^2
resume = False  # resume from previous checkpoint?

def preprop(image):
    """ prepro 210x160x3 uint8 frame into a 80x80 black/white matrix """
    image = cv2.cvtColor(cv2.resize(image, (80,80)), cv2.COLOR_RGB2GRAY)
    ret, image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    return image

def anneal_epsilon(current_e, init_e=1, end_e=.1, decay_factor=50000):
    if current_e > end_e:
        current_e -= (init_e - end_e) / decay_factor
    else:
        current_e = end_e
    return current_e

def discount_reward(reward, Q):
    """Returns the discounted reward for a given reward"""
    GAMMA = 0.99  # discount factor for reward
    return reward + GAMMA * np.max(Q)

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

def trainNetwork(sess, game_env, neural_network, state, learning_rate=1e-4):
    """Trains a neural network to play a game"""
   
    # Set of legal game actions
    num_actions = game_env.action_space.n

    # Tensorflow Placeholders to feed into neural_network and train_step
    actions = tf.placeholder(tf.float32, [None, num_actions], name="actions_pl")
    y = tf.placeholder(tf.float32, [None])

    # Get an action from the NN
    get_action = tf.reduce_sum(tf.mul(neural_network, actions), reduction_indices=1)
    
    # Default cost function is a simple root-mean squared
    cost = tf.reduce_mean(tf.square(y - get_action))
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(cost)

    # Initialize replay memory D to capacity N
    replay_memory = deque()

    # saving and loading networks
    saver = tf.train.Saver(max_to_keep=10)
    sess.run(tf.initialize_all_variables())
    checkpoint = tf.train.get_checkpoint_state("saved_networks")

    if checkpoint and checkpoint.model_checkpoint_path:
        saver.restore(sess, checkpoint.model_checkpoint_path)
        print "Successfully loaded:", checkpoint.model_checkpoint_path
    else:
        print "Could not find old network weights"

    time_step = 0
    checkpoint_global_step = 0
    epsilon = INIT_EPSILON

    for episode in range(NUM_EPISODES):
        # get the first state by doing nothing and preprocess the image to 80x80x1
        game_env.reset()
        observation, reward, done, info = game_env.step(0)
        observation = preprop(observation)
        state_t = np.stack((observation,
                               observation,
                               observation,
                               observation), axis=2)

        action_t = np.zeros(num_actions)

        while not done:
            if random.random() <= epsilon or t < 1000:
                chosen_action = game_env.action_space.sample()
            else:
                chosen_action = np.argmax(neural_network.eval(feed_dict={state: [state_t]})[0])

            # Running the network forward is computationally expensive and we don't have to make
            #  a deep, calculated response to every state. Instead we can saftely skip a few frames
            #  by repeating the last move N number of times rather than ask the network what we
            #  should do constantly. This drops training time significantly.
            for _ in range(NUM_FRAMES_SKIP):
                action_t[chosen_action] = 1 # One-hot encoding for actions
                observation_t1, reward, done, info = game_env.step(chosen_action)
                #plt.imsave("../images/pong_play/{}".format(t), observation_t1)
                observation_t1 = preprop(observation_t1).reshape((80,80,1))
                state_t1 = np.append(observation_t1, 
                                        state_t[:,:,0:3], axis=2)

                game_env.render()
                replay_memory.append((state_t,
                                      action_t,
                                      reward,
                                      state_t1,
                                      done))
                
                t += 1
            if len(replay_memory) > MEMORY_LENGTH:
                replay_memory.popleft()

            if len(replay_memory) > 500:
                # Mini batch to train on
                mini_batch = random.sample(replay_memory, BATCH_SIZE)

                # Grab the batch variables
                state_batch, action_batch, reward_batch, state_t1_batch, _ = zip(*mini_batch)
                y_batch = []

                readout_batch = neural_network.eval(feed_dict={state: state_t1_batch})
                for i in range(len(mini_batch)):
                    # If the done flag is found, reward is equal to reward of that state
                    if mini_batch[i][4]:
                        y_batch.append(reward_batch[i])
                    # Else, reward for this state is calculated from discount reward
                    else:
                        y_batch.append(discount_reward(reward_batch[i], readout_batch))

                # Take what we've gathered from our memory of past plays, and train on it
                train_step.run(feed_dict={y: y_batch,
                                          actions: action_batch,
                                          state: state_batch})
            state_t = state_t1
            epsilon = anneal_epsilon(epsilon, init_e=INIT_EPSILON)

            if t % 10000 == 0:
                saver.save(sess, 'saved_networks/pong', global_step=checkpoint_global_step)
                print " --- Running Progress -- "
                print "Episode #: ", episode
                print "Global Step: ", checkpoint_global_step
                print "Epsilon:   ", epsilon
                t = 0
                checkpoint_global_step += 1

        with open('saved_networks/reward_tracker.csv', 'a') as reward_tracker:
            total_num_rewards = sum([memory[2] for memory in replay_memory]) 
            avg_num_rewards = sum([memory[2] for memory in replay_memory]) / float(episode + 1)
            reward_tracker.write("{}, {}, {}, {}\n".format(episode, avg_num_rewards, total_num_rewards, epsilon))

        print "Total Number of Rewards: ", total_num_rewards
        print "Average Number of Rewards: ", avg_num_rewards
        print "Epsilon: ", epsilon


def lets_play():
    # Initialize Enviroment
    sess = tf.InteractiveSession()
    env = gym.make('Pong-v0')
    s, nn_readout, h_fc1 = createNetwork(env.action_space.n)
    trainNetwork(sess, env, nn_readout, s, learning_rate=learning_rate)

def main():
    lets_play()

if __name__ == "__main__":
    main()
