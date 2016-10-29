import TactileManager as tm
import BumpManager as bm
import matplotlib.pyplot as plt

def run_handcrafted_controller():
    starting_position = 0
    t_man = tm.TactileManager(starting_position, 7, tm.ControllerType.HAND_CRAFTED)
    b_man = bm.BumpManager(0, 7, 1)
    run_controller(t_man, b_man)

def run_random_controller():
    starting_position = 0
    t_man = tm.TactileManager(starting_position, 7, tm.ControllerType.RANDOM)
    b_man = bm.BumpManager(0, 7, 1)
    run_controller(t_man, b_man)

def run_q_controller():
    starting_position = 0
    t_man = tm.TactileManager(starting_position, 7, tm.ControllerType.Q_LEARNING)
    b_man = bm.BumpManager(0, 7, 1)
    run_controller(t_man, b_man)

def run_controller(t_man, b_man):
    dists = list()
    for trial in range(0, 3000):
        dist_moved = 0.0
        while (t_man.perform_action(b_man.get_location())):
            # Move the bump
            b_man.get_next_bump()
        dist_moved = t_man.get_dist_moved()
        dists.append(dist_moved)
        t_man.reset()
        b_man.reset()
    plt.plot(dists)
    print "min: " + str(min(dists))
    print "max: " + str(max(dists))
    print "mean: " + str(reduce(lambda x, y: x + y, dists) / len(dists))
    plt.show()

if __name__ == "__main__":
    run_q_controller()
