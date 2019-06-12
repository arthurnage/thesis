import matplotlib.pyplot as plt
import numpy as np
from shard.shard import Shard


class DCShard(Shard):
    def __init__(self, shard_id, detector_classifier, initial_size=100):
        self.id = shard_id
        self.dc = detector_classifier
        self.count = 0
        self.status = 0
        self.initial_size = initial_size
        self.detections = []
        self.predictions = []
        self.target = []
        self.X_train = []
        self.y_train = []
        # self.values = []

    def process_value(self, value):
        x = value[0]
        y = value[1]
        if self.count < self.initial_size:
            self.X_train.append(x)
            self.y_train.append(y)
            if self.count + 1 == self.initial_size:
                self.dc.fit(np.array(self.X_train), np.array(self.y_train))
        else:
            x = x.reshape(1, -1)
            y = y.reshape(1, -1).ravel()
            pred = self.dc.predict(x)
            self.target.append(y)
            self.predictions.append(pred)
            self.status = self.dc.partial_fit(x, y)
            if self.status is True:
                self.detections.append(self.count + 1)
        self.count += 1
        return self.status
    
    # the following code is for parallel execution experiments

    # def get_value(self, value):
    #     self.values.append(value)
    
    # def process(self):
    #     for value in self.values:
    #         self.process_value(value)
    
    def update_status(self):
        self.status = self.dc.change_detected

    def plot_confidence(self, w=500):
        pr = np.array(self.predictions).reshape(1, -1)[0]
        tr = np.array(self.target).reshape(1, -1)[0]
        acc_run = np.convolve((pr == tr) * 1, np.ones((w,)) / w, 'same')
        plt.plot(acc_run, 'r', label=f'accuracy for window with size {w}')
        plt.legend(loc='upper left')
        plt.show()
