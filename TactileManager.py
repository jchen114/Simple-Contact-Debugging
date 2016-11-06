import math
import random
import theano
import lasagne
import theano.tensor as T
import numpy as np
import random
import pickle


class ControllerType():
    HAND_CRAFTED = "handcrafted"
    RANDOM = "random"
    Q_NET_LEARNING = "QLearning"
    Q_TABLE_LEARNING = "QTableLearning"
    DBL_Q_TBL_LEARNING = "DoubleQTableLearning"


class TactileManager():

    def __init__(self, initial_state, num_discrete_states, mode = ControllerType.HAND_CRAFTED, train=True, iterations=10000):
        # Set initial position
        self.init_pos = initial_state
        self.position = initial_state
        self.bin_length = 3 / float(num_discrete_states)
        self.mode = mode
        self.num_discrete_states = num_discrete_states
        self.rewards = 0
        self.prev_state = -1
        self.a = -1
        self.reward_lb = 0.0
        # Assume mean and variance is 7 and 1
        self.actions = [(a - math.floor(num_discrete_states/2)) * self.bin_length + 7 for a in range (0, num_discrete_states)]
        if self.mode == ControllerType.Q_NET_LEARNING:
            self.QLearnerNet = QLearnerNetwork(1, 7, [128, 128], iterations)
        if self.mode == ControllerType.Q_TABLE_LEARNING:
            self.QTable = QLearnerTable(0.2, 0.9, 1.0, train, reward_lb=self.reward_lb)
        if self.mode == ControllerType.DBL_Q_TBL_LEARNING:
            self.DblQTable = DoubleQLearningTable(0.1, 0.0, 1.0, 30000)

    def set_train_mode(self, train):
        if self.mode == ControllerType.Q_TABLE_LEARNING:
            self.QTable.train = train
        if self.mode == ControllerType.DBL_Q_TBL_LEARNING:
            self.DblQTable.train = train


    def perform_action(self, location_of_bump):

        rel_pos = location_of_bump - self.position

        if rel_pos <= -1.5 or rel_pos >= 1.5:
            r = self.reward_lb
            s_n = 0
            if rel_pos < -1.5:
                s_n = -1
            elif rel_pos > 1.5:
                s_n = 7
            if self.mode == ControllerType.Q_NET_LEARNING:
                self.QLearnerNet.add_experience(self.prev_state, self.a, r, s_n)
            if self.mode == ControllerType.Q_TABLE_LEARNING:
                self.QTable.add_exp(self.prev_state, self.a, r, s_n,)
            if self.mode == ControllerType.DBL_Q_TBL_LEARNING:
                self.DblQTable.update(self.prev_state, self.a, r, s_n)
            self.rewards += r
            return False

        dist_to_move = 0.0

        # bin the relative position
        bin = math.floor((rel_pos + 1.5) / float(self.bin_length))
        self.bin = int(bin)

        if self.prev_state != -1:
            r = self.get_reward()
            s_n = self.bin
        else:
            r = 0

        if self.mode == ControllerType.HAND_CRAFTED:
            # Handcrafted action
            dist_to_move = self.actions[self.bin]
            self.prev_state = self.bin
        elif self.mode == ControllerType.Q_NET_LEARNING:
            # Q Net Learner
            if self.prev_state != -1:
                self.QLearnerNet.add_experience(self.prev_state, self.a, r, s_n)
            self.prev_state = self.bin
            self.a = self.QLearnerNet.query_network(self.bin)
            dist_to_move = self.actions[self.a]
        elif self.mode == ControllerType.Q_TABLE_LEARNING:
            # Q Table Learner
            if self.prev_state != -1:
                self.QTable.add_exp(self.prev_state, self.a, r, s_n)
            self.prev_state = self.bin
            self.a = self.QTable.get_action(self.bin)
            dist_to_move = self.actions[self.a]
        elif self.mode == ControllerType.DBL_Q_TBL_LEARNING:
            if self.prev_state != -1:
                self.DblQTable.update(self.prev_state, self.a, r, s_n)
            self.prev_state = self.bin
            self.a = self.DblQTable.get_action(self.bin)
            dist_to_move = self.actions[self.a]
        elif self.mode == ControllerType.RANDOM:
            # Random Choice
            dist_to_move = random.choice(self.actions)

        self.position += dist_to_move
        self.rewards += r

        return True

    def get_model(self):
        if self.mode == ControllerType.Q_NET_LEARNING:
            return self.QLearnerNet

    def get_reward(self):
        trn = self.bin - 3
        x = trn * self.bin_length
        #return -1 / pow(1.5, 2) * pow(x, 2) + 1
        # if self.bin == 3:
        #     return 1.5
        # elif self.bin == -1 or self.bin == 7:
        #     return self.reward_lb
        # else:
        #     return 0.0
        # if x >= 0:
        #     return -5 / 1.5 * x + 5
        # else:
        #     return 5 / 1.5 * x + 5
        if x >= 0:
            return 1/(x + 0.1)
        else:
            return -1/(x - 0.1)

    def get_dist_moved(self):
        return self.position - self.init_pos

    def reset(self):
        # randomly sample initial position
        self.position = 3 * np.random.random_sample() - 1.5
        self.prev_state = -1
        self.rewards = 0

    def save(self):
        if self.mode == ControllerType.Q_TABLE_LEARNING:
            self.QTable.save()

    def load(self):
        if self.mode == ControllerType.Q_TABLE_LEARNING:
            self.QTable.load()


class QLearnerTable():
    Qtable = {}
    Exp = {}
    num_experiences = 0
    train = True
    def __init__(self, learning_rate, discount_factor, epsilon, anneal_factor=100, num_actions=7, train=True, reward_lb=0.1):
        self.Qtable = {}
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.num_actions = num_actions
        self.anneal_factor = anneal_factor
        self.train = train
        self.num_experiences = 0
        self.reward_lb=reward_lb

    def add_exp(self, s, a, r, s_n):
        if self.train:
            if s not in self.Qtable:
                self.Qtable[s] = dict()
                self.Qtable[s] = dict.fromkeys(range(0,self.num_actions), 0)
            if s_n not in self.Qtable: # Add entry
                self.Qtable[s_n] = dict()
                if s_n == -1 or s_n == 7: # Terminal states
                    self.Qtable[s_n] = dict.fromkeys(range(0, self.num_actions), self.reward_lb)
                else:
                    self.Qtable[s_n] = dict.fromkeys(range(0, self.num_actions), 0.0)
            max_a_n = max(self.Qtable[s_n].itervalues(), key=self.Qtable[s_n].get)
            self.Qtable[s][a] = self.Qtable[s][a] + self.learning_rate * (r + self.discount_factor * max_a_n - self.Qtable[s][a])
            self.num_experiences += 1

    def get_action(self, s):
        a = 3
        if self.train:
            # Look up the action that is less explored
            # if s not in self.Exp:
            #     self.Exp[s] = dict.fromkeys(range(0, self.num_actions), 0)
            # a = min(self.Exp[s].iterkeys(), key=self.Exp[s].get)
            # self.Exp[s][a] += 1 # Utilize that action and add to it
            # a = random.choice(range(0, self.num_actions))
            # anneal epsilon
            if self.num_experiences % self.anneal_factor == 0:
                self.epsilon = (0.1 - 1.0) / 300000 * self.num_experiences + 1.0

            if self.epsilon < 0.1:
                self.epsilon = 0.1

            p = random.uniform(0, 1.0)
            if p < self.epsilon:
                a = random.choice(range(0, self.num_actions))
            else:
                a = random.choice(range(0, self.num_actions))
                if bool(self.Qtable): # Check if dict is empty
                    if s in self.Qtable:
                        # Lookup best action from table
                        a = max(self.Qtable[s].iterkeys(), key=self.Qtable[s].get)
                    else: # s does not exist in the dict
                        self.Qtable[s] = dict.fromkeys(range(0, self.num_actions), 0)
        else:
            # Lookup best action from table
            a = max(self.Qtable[s].iterkeys(), key=self.Qtable[s].get)
        return a

    def load(self):
        try:
            self.Qtable = pickle.load(open("QTable.p", "rb"))
        except IOError as e:
            print "I/O error({0}): {1}".format(e.errno, e.strerror)

    def save(self):
        pickle.dump(self.Qtable, open("QTable.p", "wb"))


class DoubleQLearningTable():

    QTables = [{}, {}]

    def __init__(self, learning_rate, discount_factor, epsilon, max_exp=100, num_actions=7, train=True, reward_lb=0.1):
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.num_actions = num_actions
        self.max_exp = max_exp
        self.train = train
        self.num_experiences = 0
        self.reward_lb = reward_lb

    def get_action(self, s):

        if self.train:

            self.check_table_entries(self.QTables[0], s)
            self.check_table_entries(self.QTables[1], s)

            self.epsilon = max((0.1 - 1.0)/self.max_exp * self.num_experiences + 1.0, 0.1)
            p = random.uniform(0, 1.0)
            if p < self.epsilon:
                # Explore randomly
                a = random.choice(range(0, self.num_actions))
            else:
                # Query both tables for the max a
                a = self.query_for_action(s)
            return a
        else:
            # Query both tables for the max a
           return self.query_for_action(s)

    def query_for_action(self, s):
        # Choose action based on QTables
        a_1 = keywithmaxval(self.QTables[0][s])
        max_a_1 = self.QTables[0][s][a_1]

        a_2 = keywithmaxval(self.QTables[1][s])
        max_a_2 = self.QTables[1][s][a_2]

        if max_a_1 >= max_a_2:
            a = a_1
        else:
            a = a_2
        return a

    def update(self, s, a, r, s_n):
        if self.train:
            table = random.choice([0,1])
            max_a_n = 0.0
            QTableToUpdate = None
            QTableToQuery = None
            if table == 0: # Update table 1 with max of table 2
                QTableToUpdate = self.QTables[0]
                QTableToQuery = self.QTables[1]
            elif table == 1: # Update table 2 with max of table 1
                QTableToUpdate = self.QTables[1]
                QTableToQuery = self.QTables[0]
            self.check_table_entries(QTableToQuery, s_n)
            max_a_n = max(QTableToQuery[s_n].itervalues(), key=QTableToQuery[s_n].get)
            self.update_table(QTableToUpdate, s, a, r, max_a_n)
            self.num_experiences += 1

    def check_table_entries(self, QTable, s):
        if s not in QTable:
            QTable[s] = dict.fromkeys(range(0, self.num_actions), 0)

    def update_table(self, QTable, s, a, r, max_a):
        if self.train:
            self.check_table_entries(QTable, s)
            QTable[s][a] = QTable[s][a] + self.learning_rate * (r + self.discount_factor * max_a - QTable[s][a])

    def print_actions(self):
        for s in range (0, 7):
            a_1 = keywithmaxval(self.QTables[0][s])
            a_2 = keywithmaxval(self.QTables[1][s])
            print "s: " + str(s) + " a1 = " + str(a_1) + " val1 = " + str(self.QTables[0][s][a_1]) + " a2 = " + str(a_2) + " val1 = " + str(self.QTables[1][s][a_2])

    def load(self):
        try:
            self.QTables[0] = pickle.load(open("QTable1.p", "rb"))
            self.QTables[1] = pickle.load(open("QTable2.p", "rb"))
        except IOError as e:
            print "I/O error({0}): {1}".format(e.errno, e.strerror)

    def save(self):
        pickle.dump(self.QTables[0], open("QTable1.p", "wb"))
        pickle.dump(self.QTables[1], open("QTable2.p", "wb"))



def keywithmaxval(d):
    """ a) create a list of the dict's keys and values;
        b) return the key with the max value"""
    v = list(d.values())
    k = list(d.keys())
    return k[v.index(max(v))]


class QLearnerNetwork():

    experiences = list()

    def __init__(self,
                 input_size,
                 output_size,
                 hidden_units,
                 train_iterations=50000,
                 eps = 1.0,
                 batch_size = 10,
                 discount_factor = 0.0,
                 reg_factor = 0.0,
                 lr = 0.001,
                 train=True):

        self.batch_size = 10
        self.init_eps = eps
        self.epsilon = eps
        self.iterations = train_iterations
        self.num_experiences = 0
        self.train = train

        state_length = 1
        # data types for model
        State = T.dmatrix("State")
        State.tag.test_value = np.random.rand(batch_size, state_length)

        ResultState = T.dmatrix("ResultState")
        ResultState.tag.test_value = np.random.rand(batch_size, state_length)

        Reward = T.col("Reward")
        Reward.tag.test_value = np.random.rand(batch_size, 1)

        Action = T.icol("Action")
        Action.tag.test_value = np.zeros((batch_size, 1), dtype=np.dtype('int32'))

        # create 2 separate neural network
        l_inA = lasagne.layers.InputLayer((None, state_length), State)
        l_inB = lasagne.layers.InputLayer((None, state_length), State)
        for units in hidden_units:
            l_hiddenA = lasagne.layers.DenseLayer(l_inA, num_units=units, nonlinearity=lasagne.nonlinearities.rectify)
            l_hiddenB = lasagne.layers.DenseLayer(l_inB, num_units=units, nonlinearity=lasagne.nonlinearities.rectify)
            l_inA = l_hiddenA
            l_inB = l_hiddenB
        self._l_outA = lasagne.layers.DenseLayer(l_inA, num_units=output_size, nonlinearity=lasagne.nonlinearities.linear)
        self._l_outB = lasagne.layers.DenseLayer(l_inB, num_units=output_size, nonlinearity=lasagne.nonlinearities.linear)

        self._learning_rate = lr
        self._discount_factor = discount_factor
        self._rho = 0.95
        self._rms_epsilon = 0.005

        self._weight_update_steps = 100
        self._updates = 0

        self._states_shared = theano.shared(
            np.zeros((batch_size, state_length),
                     dtype=theano.config.floatX)
        )

        self._next_states_shared = theano.shared(
            np.zeros((batch_size, state_length),
                     dtype=theano.config.floatX)
        )

        self._rewards_shared = theano.shared(
            np.zeros((batch_size, 1), dtype=theano.config.floatX),
            broadcastable=(False, True)
        )

        self._actions_shared = theano.shared(
            np.zeros((batch_size, 1), dtype='int32'),
            broadcastable=(False, True),
            allow_downcast=True
        )

        self._q_valsA = lasagne.layers.get_output(self._l_outA, State)
        self._q_valsB = lasagne.layers.get_output(self._l_outB, ResultState)

        self._q_func = self._q_valsA[T.arange(batch_size), Action.reshape((-1,))].reshape((-1, 1))

        target = (Reward +
                  # (T.ones_like(terminals) - terminals) *
                  self._discount_factor * T.max(self._q_valsB, axis=1, keepdims=True))
        diff = target - self._q_valsA[T.arange(batch_size),
                                      Action.reshape((-1,))].reshape((-1, 1))

        loss = 0.5 * diff ** 2
        loss = T.mean(loss)

        params = lasagne.layers.helper.get_all_params(self._l_outA)

        givens = {
            State: self._states_shared,
            ResultState: self._next_states_shared,
            Reward: self._rewards_shared,
            Action: self._actions_shared,
        }

        # SGD update
        updates = lasagne.updates.rmsprop(loss, params, self._learning_rate, self._rho,
                                          self._rms_epsilon)
        # TD update
        # updates = lasagne.updates.rmsprop(T.mean(self._q_func), params, self._learning_rate * -T.mean(diff), self._rho,
        #                                      self._rms_epsilon)

        self._train = theano.function(
            [],
            [loss, self._q_valsA],
            updates=updates,
            givens=givens)

        self._q_vals = theano.function([], self._q_valsA,
                                       givens={State: self._states_shared})

        self._bellman_error = theano.function(inputs=[State, Action, Reward, ResultState], outputs=diff,
                                              allow_input_downcast=True)

    def update_target_model(self):
        all_paramsA = lasagne.layers.helper.get_all_param_values(self._l_outA)
        lasagne.layers.helper.set_all_param_values(self._l_outB, all_paramsA)

    def add_experience(self, s, a, r, s_n):
        if len(self.experiences) % self.batch_size == 0:
            # Train network
            if self.train:
                self.experiences.append((s, a, r, s_n))
                self.train_network()
        if len(self.experiences) % 500 == 0:
            self.print_network_outs()

    def query_network(self, s):
        if self.train:
            p = random.uniform(0,1.0)

            # Linearly anneal epsilon
            self.epsilon = max((0.1 - self.init_eps) / self.iterations * len(self.experiences) + self.init_eps, 0.1)

            if p < self.epsilon:
                return random.choice(range(0, 7))
            else:
                return self.predict(s)
        else:
            return self.predict(s)

    def train_network(self):
        samples = random.sample(self.experiences, self.batch_size)
        states = list()
        actions = list()
        rewards = list()
        next_states = list()
        # Construct column of states, rewards, actions
        for (s, a, r, s_n) in samples:
            states.append(s)
            actions.append(a)
            rewards.append(r)
            next_states.append(s_n)
        states = np.vstack(states)
        actions = np.vstack(actions)
        rewards = np.vstack(rewards)
        next_states = np.vstack(next_states)
        self.train(states, actions, rewards, next_states)

    def train(self, states, actions, rewards, result_states):
        self._states_shared.set_value(states)
        self._next_states_shared.set_value(result_states)
        self._actions_shared.set_value(actions)
        self._rewards_shared.set_value(rewards)

        if ((self._updates % self._weight_update_steps) == 0):
            self.update_target_model()
        self._updates += 1
        loss, _ = self._train()
        return np.sqrt(loss)

    def predict(self, state):
        q_vals = self.q_values(state)
        return np.argmax(q_vals)

    def q_values(self, state):
        # states = np.zeros((self._batch_size, self._state_length), dtype=theano.config.floatX)
        # states[0, ...] = state
        self._states_shared.set_value([[state]])
        return self._q_vals()[0]

    def save(self):
        np.savez('model.npz', *lasagne.layers.get_all_param_values(self.l_out))

    def print_network_outs(self):
        np.set_printoptions(precision=4, linewidth=1000)
        print '========================================'
        for s in range (0, 7):
            results = self.q_values(s)
            print(results)


if __name__ == "__main__":
    import BumpManager as bm
    import matplotlib.pyplot as plt
    trials = 9000
    b_man = bm.BumpManager(0, 7, 0.4)
    d_factors = list()
    rewards = list()

    for discount_factor in np.arange (0, 1, 0.02):
        d_factors.append(discount_factor)
        dists = list()
        cum_rewards = list()
        # Training
        t_man = TactileManager(0, 7, ControllerType.DBL_Q_TBL_LEARNING,train=True)
        t_man.DblQTable.discount_factor = discount_factor # Set discount factor
        b_man.reset()
        print "discount factor: " + str(discount_factor)
        for trial in range(0, trials):
            while t_man.perform_action(b_man.get_location()):
                # Move the bump
                b_man.get_next_bump()
            b_man.reset()
            t_man.reset()
        t_man.set_train_mode(False)
        for test in range (0, 50):
            dist_moved = 0.0
            while t_man.perform_action(b_man.get_location()):
                # Move the bump
                b_man.get_next_bump()
            dist_moved = t_man.get_dist_moved()
            dists.append(dist_moved)
            cum_rewards.append(t_man.rewards)
            t_man.reset()
            b_man.reset()

        print "min: " + str(min(dists))
        print "max: " + str(max(dists))
        print "std: " + str(np.std(dists))
        print "mean: " + str(np.mean(dists))
        print "rewards: " + str(np.mean(cum_rewards))
        rewards.append(np.mean(cum_rewards))

    plt.plot(d_factors, rewards)
    plt.show()
