import TactileManager as tm
import BumpManager as bm
import matplotlib.pyplot as plt
import numpy as np


def run_handcrafted_controller(mean, var, iter):
    starting_position = 0
    print("Handcrafted:")
    t_h_man = tm.TactileManager(starting_position, 7, tm.ControllerType.HAND_CRAFTED)
    b_man = bm.BumpManager(0, mean, var)
    run_controller(t_h_man, b_man, iter, False)


def run_random_controller(mean, var, iter):
    starting_position = 0
    print("Random: ")
    t_r_man = tm.TactileManager(starting_position, 7, tm.ControllerType.RANDOM)
    b_man = bm.BumpManager(0, mean, var)
    run_controller(t_r_man, b_man, iter, False)


def run_q_net_controller(mean, var, iter):
    print("Q Network: ")
    starting_position = 0
    t_n_man = tm.TactileManager(starting_position, 7, tm.ControllerType.Q_NET_LEARNING, iterations=10000)
    b_man = bm.BumpManager(0, mean, var)
    t_n_man.set_train_mode(True)
    run_controller(t_n_man, b_man, 2000, False)
    t_n_man.get_model().print_network_outs()
    t_n_man.set_train_mode(False)
    run_controller(t_n_man, b_man, iter, False)


def run_q_table_controller(mean, var, iter):
    starting_pos = 0.0
    t_t_man = tm.TactileManager(starting_pos, 7, tm.ControllerType.Q_TABLE_LEARNING)
    b_man = bm.BumpManager(0, mean, var)

    #t_t_man.load()
    print("Running Table controller")
    print("Training:")
    t_t_man.set_train_mode(train=True)
    run_controller(t_t_man, b_man, 15000, False)
    #t_t_man.save()

    print("Testing:")
    t_t_man.set_train_mode(train=False)
    run_controller(t_t_man, b_man, iter, False)


def run_dbl_q_table_learning(mean, var, iter):
    starting_pos = 0.0
    t_dt_man = tm.TactileManager(starting_pos, 7, tm.ControllerType.DBL_Q_TBL_LEARNING)
    b_man = bm.BumpManager(0, mean, var)

    #t_dt_man.load()
    print("Running Double Q Table controller")
    print("Training:")
    t_dt_man.set_train_mode(train=True)
    run_controller(t_dt_man, b_man, 8500, False)
    #t_dt_man.save()
    t_dt_man.DblQTable.print_actions()
    print("Testing:")
    t_dt_man.set_train_mode(train=False)
    run_controller(t_dt_man, b_man, iter, True)



def run_controller(t_man, b_man, trials=3000, plot=True):
    dists = list()
    cum_rewards = list()
    t_man.reset()
    b_man.reset()
    for trial in range(0, trials):
        dist_moved = 0.0
        while t_man.perform_action(b_man.get_location()):
            # Move the bump
            b_man.get_next_bump()
        dist_moved = t_man.get_dist_moved()
        dists.append(dist_moved)
        cum_rewards.append(np.sum(t_man.rewards))
        t_man.reset()
        b_man.reset()
    if plot:
        plt.plot(dists)
        plt.show()
        b_man.plot()
    print "min: " + str(min(dists))
    print "max: " + str(max(dists))
    print "std: " + str(np.std(dists))
    print "mean: " + str(np.mean(dists))
    print "rewards: " + str(np.mean(cum_rewards))


if __name__ == "__main__":
    mean = 7
    var = 1.0
    iter = 50
    #run_random_controller(mean, var, iter)
    #run_handcrafted_controller(mean, var, iter)
    #run_q_table_controller(mean, var, iter)
    #run_dbl_q_table_learning(mean, var, iter)
    run_q_net_controller(mean, var, iter)
