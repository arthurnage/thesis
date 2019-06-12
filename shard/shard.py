from abc import ABC, abstractmethod


class Shard(ABC):
    """
    Class that implements an interface for a shard
    """
    id = 0  # shard id
    status = 0  # current status (0 : no changes, 1 : changes detected)

    @abstractmethod
    def process_value(self, value):
        """Incoming value processing"""
        pass

    @abstractmethod
    def plot_confidence(self):
        """Plots history of changes in confidence level"""
        pass

    @abstractmethod
    def update_status(self):
        """Updates current status"""
        pass
