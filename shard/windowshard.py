import scipy
import numpy as np
import matplotlib.pyplot as plt

from collections import deque
from datasketches import KllFloatSketch
from thesis.shard import Shard


class WindowShard:
    """A shard that uses sliding windows for data accumulation"""

    def __init__(
            self, worker_id, big_window_size=5000, small_window_size=500,
            stat_test=scipy.stats.kstest, alpha=0.05):
        self.id = worker_id

        self.big_window = deque()
        self.small_window = deque()
        self.big_p_values = []
        self.small_p_values = []

        self.big_window_size = big_window_size
        self.small_window_size = small_window_size
        self.stat_test = stat_test
        self.alpha = alpha
        self.status = 0
        # self.fill()
        # self.test()
        self.reset_p_values()

    def process_value(self, value):
        self.small_window.appendleft(value)
        self.big_window.appendleft(self.small_window.pop())
        self.big_window.pop()

    def fill(self):
        self.big_window = deque(
            scipy.stats.norm.rvs(size=self.big_window_size))
        self.small_window = deque(
            scipy.stats.norm.rvs(size=self.small_window_size))

    def reset_p_values(self):
        self.big_p_values = []
        self.small_p_values = []

    def update_status(self):
        if self.big_p_values[-1] < self.alpha and self.small_p_values[-1] < self.alpha:
            self.status = 1

    def test(self):
        _, big_p_value = self.stat_test(self.big_window, 'norm')
        _, small_p_value = self.stat_test(self.small_window, 'norm')
        self.big_p_values.append(big_p_value)
        self.small_p_values.append(small_p_value)

    def plot_confidence(self):
        plt.plot(self.small_p_values, 'r', label='small window')
        plt.plot(self.big_p_values, 'b', label='big_window')
        plt.axhline(self.alpha)
        plt.legend(loc='upper left')
        plt.show()