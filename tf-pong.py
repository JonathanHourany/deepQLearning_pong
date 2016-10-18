import tensorflow as tf
import numpy as np
import random
import sys
import gym
import cv2
from collections import deque

def preprop(image):
    """ prepro 210x160x3 uint8 frame into a 80x80 black/white matrix """
    image = cv2.cvtColor(cv2.resize(image, (80,80)), cv2.COLOR_RGB2GRAY)
    ret, image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    return image

def anneal_epsilon(current_e, init_e=1, end_e=.1, decay_factor=50000):
    if current_e > end_e:
        current_e -= (init_e - end_e) / decay_factor
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
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

def createNetwork(actions):
    # Input Layer
    state = tf.placeholder(tf.float32, [None, 80, 80, 4], name="state_pl")

    # Hidden Layers
    W_conv1 = init_weight_matrix([8, 8, 4, 32])
    b_conv1 = init_bias_matrix([32])
    h_conv1 = tf.nn.relu(conv2d(state, W_conv1, 4) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    W_conv2 = init_weight_matrix([4, 4, 32, 64])
    b_conv2 = init_bias_matrix([64])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2, 2) + b_conv2)

    W_conv3 = init_weight_matrix([3, 3, 64, 64])
    b_conv3 = init_bias_matrix([64])
    h_conv3 = tf.nn.relu(conv2d(h_conv2, W_conv3, 1) + b_conv3)

    h_conv3_flat = tf.reshape(h_conv3, [-1, 1600])

    W_fc1 = init_weight_matrix([1600, 512])
    b_fc1 = init_bias_matrix([512])
    #out_flatten = tf.reshape(h_conv3_flat, [-1, 1600])
    h_fc1 = tf.nn.relu(tf.matmul(h_conv3_flat, W_fc1) + b_fc1)

    # Output Layer
    W_fc2 = init_weight_matrix([512, actions])
    b_fc2 = init_bias_matrix([actions])
    output = tf.matmul(h_fc1, W_fc2) + b_fc2
    
    return state, output, h_fc1

def trainNetwork(sess, game_env, neural_network, state, learning_rate=1e-4):
    """Trains a neural network to play a game"""
   
    # Set of legal game actions
    num_actions = game_env.action_space.n
    actions = tf.placeholder(tf.float32, [None, num_actions], name="actions_pl")
    y = tf.placeholder(tf.float32, [None])
    readout_action = tf.reduce_sum(tf.mul(neural_network, actions), reduction_indices=1)
    
    # Default cost function is a simple root-mean squared
    cost = tf.reduce_mean(tf.square(y - readout_action))
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

    t = 0
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
                #print "Chosen Action: ", chosen_action
            action_t[chosen_action] = 1 # One-hot encoding for actions
            observation_t1, reward, done, info = game_env.step(chosen_action)
            observation_t1 = preprop(observation_t1).reshape((80,80,1))
            state_t1 = np.append(observation_t1, 
                                    state_t[:,:,0:3], axis=2)

            game_env.render()
            replay_memory.append((state_t,
                                  action_t,
                                  reward,
                                  state_t1,
                                  done))
            
            if len(replay_memory) > MEMORY_LENGTH:
                replay_memory.popleft()

            if len(replay_memory) > 500:
                # Mini batch to train on
                mini_batch = random.sample(replay_memory, BATCH_SIZE)

                # Grab the batch variables
                #state_batch = [memory[0] for memory in mini_batch]
                #action_batch = [memory[1] for memory in mini_batch]
                #reward_batch = [memory[2] for memory in mini_batch]
                #state_t1_batch = [memory[3] for memory in mini_batch]
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
            #state_t = state_t1
            t += 1
            state_t = state_t1
            epsilon = anneal_epsilon(epsilon, init_e=INIT_EPSILON)

            if t % 10000 == 0:
                saver.save(sess, 'saved_networks/pong', global_step=t)
                print " --- Running Progress -- "
                print "Episode #: ", episode
                print "Time Step: ", t
                print "Epsilon:   ", epsilon

        with open('saved_networks/reward_tracker.csv', 'a') as reward_tracker:
            total_num_rewards = sum([memory[2] for memory in replay_memory]) 
            avg_num_rewards = sum([memory[2] for memory in replay_memory]) / float(episode + 1)
            reward_tracker.write("{}, {}, {}, {}\n".format(episode, avg_num_rewards, total_num_rewards, epsilon))

        print "Total Number of Rewards: ", total_num_rewards
        print "Average Number of Rewards: ", avg_num_rewards

## Hyperparameters
learning_rate = 1e-1
MEMORY_LENGTH = 1000000
NUM_EPISODES = 5000000
INIT_EPSILON = 1
BATCH_SIZE = 32
decay_rate = 0.99  # decay factor for RMSProp leaky sum of grad^2
resume = False  # resume from previous checkpoint?
render = False

def lets_play():
    # Initialize Enviroment
    sess = tf.InteractiveSession()
    env = gym.make('Pong-v0')
    s, nn_readout, h_fc1 = createNetwork(env.action_space.n)
    trainNetwork(sess, env, nn_readout, s)

def main():
    lets_play()

if __name__ == "__main__":
    main()

# for episode in range(NUM_EPISODES):
#     observation = env.reset()
#     for step in range(100):
#         env.render()
#         action = env.action_space.sample()
#         observation, reward, done, info = env.step(action)
#         print reward, action, episode
#         if done:
#             print "Episode finshed after {} timesteps".format(step + 1)
#             break