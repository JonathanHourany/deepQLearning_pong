from collections import deque
import tensorflow as tf
import numpy as np
import random
import sys
import cv2

class DeepQLearner(object):
    """An instance of a DQN"""

    def __init__(self, game_env=None, neural_network=None, learning_rate=1e-4, gamma=.99, replay_len=1000000, min_replay_len=1000,
                 epsilon=1, min_epsilon=0.1, anneal_epsilon=True, anneal_rate=500, skip_n_frames=4):

        self._game_env = game_env
        self._lr = learning_rate
        self._gamma = gamma
        self._neural_network = neural_network
        self._epsilon = epsilon
        self._init_epsilon = epsilon
        self._min_epsilon = min_epsilon
        self._anneal_rate = anneal_rate
        self._skip_n_frames = skip_n_frames
        self._replay_memory = deque(maxlen=replay_len)
        self._min_replay_len = min_replay_len
        self.anneal_epsilon = anneal_epsilon
        self.checkpoint_path = "saved_networks/pong/"

    def _initialize_new_game(self):
        self._game_env.reset()
        observation, reward, done, info = self._game_env.step(0)
        observation = self.preproc_image(observation)
        state_t = np.stack((observation,
                            observation,
                            observation,
                            observation), axis=2)

        return state_t, np.zeros(self.action_space_num), done

    def preproc_image(self, image):
        """ Preprocess an uint8 frame into a 80x80 black/white matrix """
        image = cv2.cvtColor(cv2.resize(image, (80, 80)), cv2.COLOR_RGB2GRAY)
        ret, image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
        return image

    def resume_checkpoint(self, path):
        pass

    def attach_game_env(self, env):
        pass

    def discount_reward(self, reward, Q):
        """Returns the discounted reward for a given reward"""
        GAMMA = 0.99  # discount factor for reward
        return reward + GAMMA * np.max(Q)

    def _anneal_epsilon(self):
        if self.anneal_epsilon and (self._epsilon > self._min_epsilon):
            self._epsilon -= (self._init_epsilon - self._min_epsilon) / float(self._anneal_rate)
        else:
            self._epsilon = self._min_epsilon

    def train_network(self, sess, state, batch_size=32, save=True, resume_from_checkpoint=True, max_to_keep=6):
        # Set the number of legal actions available in game
        try:
            self.action_space_num = self._game_env.action_space.n
        except AttributeError:
            print "An OpenAI Gym game enviroment must be set first"

        #######
        # Variables used in training the neural net

        # Tensorflow Placeholders to feed into neural_network and train_step
        actions = tf.placeholder(tf.float32, [None, self.action_space_num], name="actions_pl")
        y = tf.placeholder(tf.float32, [None], name='y_pl')

        # Matrix multiplies a batch of actions the NN would take given a state S against
        # against a batch of actions that were taken during the same state
        get_actions = tf.reduce_sum(tf.mul(self._neural_network, actions), reduction_indices=1)

        # RMSE Cost Function
        cost = tf.reduce_mean(tf.square(y - get_actions))
        train_step = tf.train.AdamOptimizer(self._lr).minimize(cost)
        #######

        if resume_from_checkpoint:
            saver = tf.train.Saver(max_to_keep=max_to_keep)
            sess.run(tf.initialize_all_variables())
            checkpoint = tf.train.get_checkpoint_state(self.checkpoint_path)
            try:
                saver.restore(sess, checkpoint.model_checkpoint_path)
                print "Checkpoint Sucessfully Restored from ", checkpoint.model_checkpoint_path
            except: # TODO: Catch specific exception raised from checkpoint.model_checkpoint_path
                print "Could not find checkpoint. Weights will not be initialized"

        # Train/Play games until cows come home/AWS is blue in the face/meteor crashes into Earth/etc
        while True:
            state_t, action_t, done = self._initialize_new_game()
            t = 0
            episode = 0
            checkpoint_global_step = 0

            # OpenAI Gym returns a true/false 'done' state with every state
            while not done:
                if len(self._replay_memory) < self._min_replay_len or random.random() <= self._epsilon:
                    chosen_action = self._game_env.action_space.sample()
                else:
                    chosen_action = np.argmax(self._neural_network.eval(feed_dict={state: [state_t]})[0])

                # Running the network forward is computationally expensive and we don't have to make
                #  a deep, calculated response to every state. Instead we can saftely skip a few frames
                #  by repeating the last move N number of times rather than ask the network what we
                #  should do constantly. This drops training time significantly.
                for _ in range(self._skip_n_frames):
                    action_t[chosen_action] = 1  # One-hot encoding for actions
                    observation_t1, reward, done, info = self._game_env.step(chosen_action)
                    #plt.imsave("../images/pong_play/{}".format(t), observation_t1)
                    observation_t1 = self.preproc_image(observation_t1).reshape((80, 80, 1))
                    state_t1 = np.append(observation_t1,
                                         state_t[:, :, 0:3], axis=2)

                    self._game_env.render()
                    self._replay_memory.append((state_t,
                                               action_t,
                                               reward,
                                               state_t1,
                                               done))

                if len(self._replay_memory) > self._min_replay_len:
                    # Mini batch to train on
                    mini_batch = random.sample(self._replay_memory, batch_size)

                    # Grab the batch variables from replay_memory
                    state_batch, action_batch, reward_batch, state_t1_batch, done_batch = zip(*mini_batch)
                    y_batch = []

                    readout_batch = self._neural_network.eval(feed_dict={state: state_t1_batch})
                    for i in range(len(mini_batch)):
                        # If the done flag is found, reward is equal to reward of that state
                        if done_batch[i]:
                            y_batch.append(reward_batch[i])
                        # Else, reward for this state is calculated from discount reward
                        else:
                            y_batch.append(self.discount_reward(reward_batch[i], readout_batch))

                    # Take what we've gathered from our memory of past plays, and train on it
                    train_step.run(feed_dict={y: y_batch,
                                              actions: action_batch,
                                              state: state_batch})

                # What was the 'next' state, becomes the current
                state_t = state_t1
                # Reduce the probability that a random action next state
                self._anneal_epsilon()

            if t % 10000 == 0:
                saver.save(sess, 'saved_networks/pong/pong-', global_step=checkpoint_global_step)
                print " --- Running Progress -- "
                print "Episode #: ", episode
                print "Global Step: ", checkpoint_global_step
                print "Epsilon:   ", self._epsilon
                t = 0
                checkpoint_global_step += 1

            with open(self.checkpoint_path + 'reward_tracker.csv', 'a') as reward_tracker:
                total_num_rewards = sum([memory[2] for memory in self._replay_memory])
                avg_num_rewards = sum([memory[2] for memory in self._replay_memory]) / float(episode + 1)
                reward_tracker.write("{}, {}, {}, {}\n".format(episode, avg_num_rewards, total_num_rewards, self._epsilon))

            episode += 1
            print "Total Number of Rewards: ", total_num_rewards
            print "Average Number of Rewards: ", avg_num_rewards
            print "Epsilon: ", self._epsilon

    def feed_network(self, **kwargs):
        pass
