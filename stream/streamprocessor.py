import numpy as np
import matplotlib.pyplot as plt
import time

# for parallel execution
# from multiprocessing import Process, Queue


class StreamProcessor:
    def __init__(self, shard, shard_args, partitioner, generator, n_shards=1):
        self.n_shards = n_shards
        self.shards = [shard(i, *shard_args) for i in range(n_shards)]
        self.partitioner = partitioner
        self.generator = generator
        self.shards_status = [0 for w in self.shards]
        self.status_history = []
        self.counter = 0
        self.info = []

        # for parallel execution
        # self.processes = [Process(target=self.shards[i].process) for i in range(n_shards)] 

    def send_values(self, n=1000):
        values = self.generator.generate_values()
        for i in range(len(values)):
            self.counter += 1
            ind = self.partitioner.make_partition(values[i])
            # self.shards[ind].get_value(values[i])  # for parallel execution
            self.shards[ind].process_value(values[i])
            if i % self.n_shards == 0:
                self.update_status()
                self.status_history.append(np.sum(self.shards_status))

    # for parallel execution
    # def process(self):
    #     start = time.time()
    #     for p in self.processes:
    #         p.start()
    #     for p in self.processes:
    #         p.join()
    #     end = time.time()
    #     print(end - start)

    def update_status(self):
        for w in self.shards:
            w.update_status()
        self.shards_status = [w.status for w in self.shards]

    def plot_history(self):
        plt.plot(self.status_history, 'r')
        plt.show()
