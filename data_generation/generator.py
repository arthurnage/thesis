import numpy as np
import random

from abc import ABC, abstractmethod


class Generator(ABC):
    """
    Class that implements data generation
    """
    @abstractmethod
    def generate_values(self, n):
        """returns generated data"""
        pass


class NormalDistributionGenerator(Generator):
    def __init__(self, mu=0, sigma=1):
        self.mu = mu
        self.sigma = sigma

    def generate_values(self, n=1000, noise_id=-1):
        values = np.zeros(n)
        for i in range(n):
            mu = self.mu
            uid = random.randint(0, 100)
            if uid == noise_id:
                mu = 100
            values[i] = random.gauss(mu, self.sigma), uid
        return values


class UniformDistributionGenerator(Generator):
    def __init__(self, a=-1, b=1):
        self.a = a
        self.b = b

    def generate_values(self, n=1000, noise_id=-1):
        values = np.zeros(n)
        for i in range(n):
            a = self.a
            uid = random.randint(0, 100)
            if uid == noise_id:
                a = -100
            values[i] = random.uniform(a, self.b), uid
        return values

class DFReader(Generator):
    def __init__(self, df):
        self.df = df
    
    def generate_values(self, n=50000):
        values = [(row[:-1], row[-1]) for row in self.df]
        return values