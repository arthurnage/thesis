import scipy
import numpy as np
import matplotlib.pyplot as plt

from thesis.shard import Shard
from datasketches import KllFloatSketch



class QSketchShard(Shard):
    """A shard that uses quantile sketch for data accumulation"""
    def __init__(self, shard_id, sketch_k=4000, stat_test=scipy.stats.kstest, alpha=0.05):
        self.id = shard_id
        self.sketch_k = sketch_k
        self.__reset_sketch()
        self.stat_test = stat_test
        self.alpha = alpha
        self.trash = None
        self.p_values = []
        self.status = 0
        self.counter = 0

    def process_value(self, value, repeat_test=1, test=True):
        self.counter += 1
        self.sketch.update(value)
        if test and self.counter % repeat_test == 0:
            self.test()

        self.update_status()
        return self.status

    def __reset_sketch(self):
        self.sketch = KllFloatSketch(self.sketch_k)

    def __reset_trash(self):
        self.trash = self.counter + 1

    def sketch_values(self):
        m = self.sketch.getNumRetained()
        values = self.sketch.getQuantiles(list(np.arange(m) / m))
        return values

    def sketch_mean(self):
        return np.mean(self.sketch_values())

    def sketch_std(self):
        return np.std(self.sketch_values())

    def update_status(self):
        if self.p_values[-1] < self.alpha:
            self.status = 1

    def test(self):
        m = self.sketch.getNumRetained()
        values = self.sketch.getQuantiles(list(np.arange(m) / m))
        values = values - self.sketch_mean()
        values = values / self.sketch_std()
        _, p_value = self.stat_test(values, 'norm')

        self.p_values.append(p_value)

    def plot_confidence(self):
        plt.plot(self.p_values, 'r', label='p values')
        plt.axhline(self.alpha)
        plt.legend(loc='upper left')
        plt.show()
