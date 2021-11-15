import torch
from abc import ABC, abstractmethod


class BaseMetric(ABC):
    """Base metric declaration"""
    @abstractmethod
    def __init__(self):
        """Initialize metric"""
        pass

    @abstractmethod
    def update(self):
        """Update info for calculating metric.
        """
        pass

    @abstractmethod
    def reset(self):
        """Reset metric"""
        pass

    @abstractmethod
    def compute(self):
        """Compute metric"""
        pass
