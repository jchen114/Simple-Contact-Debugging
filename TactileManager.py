import math
import random
import theano
import lasagne
import theano.tensor as T
import numpy as np
import random

class ControllerType():
    HAND_CRAFTED = "handcrafted"
    RANDOM = "random"
    Q_LEARNING = "QLearning"


class TactileManager():

    def __init__(self, initial_state, num_discrete_states, mode = ControllerType.HAND_CRAFTED):
        # Set initial position
        self.init_pos = initial_state
        self.position = initial_state
        self.bin_length = 3 / float(num_discrete_states)
        self.mode = mode
        self.num_discrete_states = num_discrete_states
        # Assume mean and variance is 7 and 1
        self.actions = [5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 9.0]
        if self.mode == ControllerType.Q_LEARNING:
            self.QLearner = QLearner(1, 7, [128, 128])

    def perform_action(self, location_of_bump):

        rel_pos = location_of_bump - self.position

        if rel_pos < -1.5 or rel_pos > 1.5:
            return False

        dist_to_move = 0.0

        # bin the relative position
        bin = math.floor((rel_pos + 1.5) / float(self.bin_length))
        self.bin = int(bin)

        if self.mode == ControllerType.HAND_CRAFTED:
            # Handcrafted action
            dist_to_move  = self.actions[bin]
        elif self.mode == ControllerType.Q_LEARNING:
            # Q Learner
            a = self.QLearner.query_network(self.bin)
            dist_to_move = self.actions[a]
        else:
            # Random Choice
            dist_to_move = random.choice(self.actions)

        self.position += dist_to_move
        return True

    def get_dist_moved(self):
        return self.position - self.init_pos

    def reset(self):
        self.position = self.init_pos

class QLearner():

    experiences = list()

    def __init__(self, input_size, output_size, hidden_units, batch_size = 8, discount_factor = 0.9, reg_factor = 0.0, lr = 0.001):

        x = T.dmatrix('x')
        y = T.dmatrix('y')

        self.batch_size = batch_size
        self.discount_factor = discount_factor

        l_in = lasagne.layers.InputLayer((None, input_size), input_var=x)
        for units in hidden_units:
            l_hidden = lasagne.layers.DenseLayer(l_in, num_units=units, nonlinearity=lasagne.nonlinearities.rectify)
            l_in = l_hidden
        l_out = lasagne.layers.DenseLayer(l_in, num_units=output_size, nonlinearity=lasagne.nonlinearities.linear)
        self.l_out = l_out
        params = lasagne.layers.get_all_params(l_out, trainable=True)
        cost = T.mean(0.5 * (y - lasagne.layers.get_output(l_out)))
        updates = lasagne.updates.rmsprop(cost, params, lr, 0.95, 0.001)

        try:
            with np.load('model.npz') as f:
                param_values = [f['arr_%d' % i] for i in range(len(f.files))]
            lasagne.layers.set_all_param_values(l_out, param_values)
        except IOError as e:
            print "I/O error({0}): {1}".format(e.errno, e.strerror)

        self.n_train = theano.function(
            inputs=[x, y],
            outputs=[lasagne.layers.get_output(l_out), cost],
            updates=updates
        )

        self.n_predict = theano.function(
            inputs=[x],
            outputs=[lasagne.layers.get_output(l_out)]
        )

    def add_experience(self, s, a, r, s_n):
        self.experiences.append((s, a, r, s_n))
        if (len(self.experiences) % self.batch_size == 0):
            # Train network
            self.train_network()

    def query_network(self, s):
        q_values = self.n_predict([[s]])
        return q_values.index(max(q_values))

    def train_network(self):
        samples = random.sample(self.experiences, self.batch_size)
        labels = list()
        for (s, a, r, s_n) in samples:
            q_values = self.n_predict([[s]])
            q_values_n = self.n_predict([[s_n]])
            max_q = max(q_values_n)
            lbl_av = q_values
            lbl_av[a] = r + self.discount_factor * max_q - q_values[a]
            labels.append(lbl_av)
        input_mat = np.matrix(samples)
        label_mat = np.matrix(labels)
        self.n_train([input_mat, label_mat])

    def save(self):
        np.savez('model.npz', *lasagne.layers.get_all_param_values(self.l_out))


if __name__ == "__main__":
    TactileManager(0, 7)
