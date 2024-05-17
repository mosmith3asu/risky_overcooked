import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count

import numpy as np
import tensorflow as tf
from tensorflow import keras

#######################################
# Replay Memory #######################
#######################################
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'done'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


#######################################
# DQN ################################
#######################################
class DQN(object):
    """
    Reference for CNN-DQN and DDQN: https://github.com/yxu1168/Reinforcement-Learning-DQN-for-ATARI-s-Pong-Game---TensorFlow-2.0-Keras
    """

    def __init__(self, obs_shape, n_actions,
                 num_filters, num_convs,                #CNN params
                 num_hidden_layers, size_hidden_layers, #MLP params
                 learning_rate, seed,
                 **kwargs):
        # super(DQN, self).__init__()
        # ## Parse custom network params
        # # num_hidden_layers = custom_params["NUM_HIDDEN_LAYERS"]
        # # size_hidden_layers = custom_params["SIZE_HIDDEN_LAYERS"]
        # # num_filters = custom_params["NUM_FILTERS"]
        # # num_convs = custom_params["NUM_CONV_LAYERS"]
        # # cell_size = custom_params["CELL_SIZE"]
        # # learning_rate = custom_params["LEARNING_RATE"]
        # # seed = custom_params["SEED"]
        #
        # Parameters
        self.n_outputs = n_actions
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        self.loss_fn = keras.losses.mean_squared_error
        tf.random.set_seed(seed)
        np.random.seed(seed)
        #
        #
        #
        # ### Create graph of the model ###
        # obs_inputs = tf.keras.Input(shape=obs_shape, name="input")
        # out = obs_inputs
        # ####################################################
        # ## Initial CNN "vision" network ####################
        # # Apply initial conv layer with a larger kenel (why?)
        # if num_convs > 0:
        #     out = tf.keras.layers.Conv2D(
        #             filters=num_filters,
        #             kernel_size=[5, 5],
        #             padding="same",
        #             activation=tf.nn.leaky_relu,
        #             name="conv_initial",
        #         )(out)
        #
        # # Apply remaining conv layers, if any
        # for i in range(0, num_convs - 1):
        #     padding = "same" if i < num_convs - 2 else "valid"
        #     # tf.keras.layers.TimeDistributed(...
        #     out = tf.keras.layers.Conv2D(
        #             filters=num_filters,
        #             kernel_size=[3, 3],
        #             padding=padding,
        #             activation=tf.nn.leaky_relu,
        #             name="conv_{}".format(i),
        #         )(out)
        #
        # # Flatten spatial features
        # out = tf.keras.layers.TimeDistributed(tf.keras.layers.Flatten())(out)
        #
        #
        # ####################################################
        # # Apply dense hidden layers, if any ################
        # for i in range(num_hidden_layers):
        #     out = tf.keras.layers.Dense(
        #             units=size_hidden_layers,
        #             activation=tf.nn.leaky_relu,
        #             name="fc_{0}".format(i),
        #         )(out)
        #
        #
        # ####################################################
        # # Linear last layer for action quality ############
        # # layer_out = tf.keras.layers.Dense(self.num_outputs, name="logits")(out)
        # # value_out = tf.keras.layers.Dense(self.n_outputs, name="values")(out)
        # value_out = tf.keras.layers.Dense(self.n_outputs, name="values", activation=None)(out)
        #
        # # self.cell_size = cell_size
        # self.base_model = tf.keras.Model(inputs=[obs_inputs], outputs=[value_out])

        # Alternative model declaration ####################
        self.base_model = keras.models.Sequential([
            keras.layers.Conv2D(filters=num_filters, kernel_size=[5, 5], padding="same", activation=tf.nn.leaky_relu,  name="conv_initial"),
            keras.layers.Conv2D(filters=num_filters, kernel_size=[3, 3], padding="same", activation=tf.nn.leaky_relu,  name="conv_0"),
            keras.layers.Conv2D(filters=num_filters, kernel_size=[3, 3], padding="valid", activation=tf.nn.leaky_relu, name="conv_1"),
            keras.layers.Flatten(),
            keras.layers.Dense(units=size_hidden_layers, activation=tf.nn.leaky_relu, name="fc_0"),
            keras.layers.Dense(units=size_hidden_layers, activation=tf.nn.leaky_relu, name="fc_1"),
            keras.layers.Dense(units=size_hidden_layers, activation=tf.nn.leaky_relu, name="fc_2"),
            keras.layers.Dense(self.n_outputs,name="values") # activation=None
        ])

        # self.base_model.summary()



    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        return self.base_model.predict(x)
    def predict(self, x):
        return self.forward(x)

    @property
    def trainable_variables(self):
        return self.base_model.trainable_variables