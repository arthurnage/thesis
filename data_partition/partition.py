import mmh3
from abc import ABC, abstractmethod


class Partitioner(ABC):
    """
    Class that implements different data partition options
    """
    @abstractmethod
    def make_partition(self, value, n):
        """returns index of a chosen shard for data"""
        pass

class RoundRobin(Partitioner):
    def __init__(self, n):
        self.n = n
        self.last_n = -1

    def make_partition(self, value):
        self.last_n = (self.last_n + 1) % self.n
        return self.last_n

class IdHasher(Partitioner):
    def __init__(self, n):
        self.n = n
    
    def make_partition(self, value):
        val, id = value[:-1], value[-1]
        return mmh3.hash(str(id)) % self.n

class ValueHasher(Partitioner):
    def __init__(self, n):
        self.n = n
    
    def make_partition(self, value):
        val, id = value[:-1], value[-1]
        return mmh3.hash(str(val)) % self.n

