import numpy as np
import math
import matplotlib.pyplot as plt

class BumpManager:

    def __init__(self, initial_start, mean, variance):
        self.mean = mean
        self.variance = variance
        self.position = initial_start
        self.initial_position = initial_start
        self.discrete_distances = variance/1.0
        self.positions = list()

    def get_location(self):
        return self.position

    def get_next_bump(self):
        # Return distance to next bump

        # p = np.random.random_sample()
        # if p < 0.1:
        #    self.position += 6
        # elif p >= 0.1 and p < 0.2:
        #    self.position += 6.5
        # elif p >= 0.2 and p < 0.8:
        #     self.position += 7
        # elif p >= 0.8 and p < 0.9:
        #     self.position += 7.5
        # elif p >= 0.9:
        #    self.position += 8

        #self.position += np.random.normal(self.mean, self.variance)

        # dist_to_travel = np.random.normal(self.mean, self.variance)
        # variance = dist_to_travel - self.mean
        # bin = round(variance/self.discrete_distances)
        # d = bin * self.discrete_distances
        # self.positions.append(self.mean + d)
        # self.position += self.mean + d

        # discretize the position

        self.position += 7

        return self.position

    def reset(self):
        self.position = self.initial_position

    def plot(self):
        # the histogram of the data
        n, bins, patches = plt.hist(self.positions, 50, facecolor='green', alpha=0.75)
        plt.grid(True)
        plt.show()