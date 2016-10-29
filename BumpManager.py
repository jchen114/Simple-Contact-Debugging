import numpy as np

class BumpManager:

    def __init__(self, initial_start, mean, variance):
        self.mean = mean
        self.variance = variance
        self.position = initial_start
        self.initial_position = initial_start

    def get_location(self):
        return self.position

    def get_next_bump(self):
        # Return distance to next bump
        self.position += np.random.normal(self.mean, self.variance)
        return self.position

    def reset(self):
        self.position = self.initial_position