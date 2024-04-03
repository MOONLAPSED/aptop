from abc import ABC, abstractmethod
import uuid
import argparse
import threading
import logging
import queue
import json
import openai
from openai import OpenAI
import os
from typing import Optional


class BaseContextManager(ABC):
    """
    Abstract base class for custom context managers.
    """
    @abstractmethod
    def __enter__(self):
        pass

    @abstractmethod
    def __exit__(self, exc_type, exc_value, traceback):
        pass

    @abstractmethod
    def _root(self):
        pass


class BaseProtocol(ABC):
    """
    Abstract base class for custom protocols.
    """
    @abstractmethod
    def send(self):
        pass

    @abstractmethod
    def get(self):
        pass

    @abstractmethod
    def post(self):
        pass

    @abstractmethod
    def publish(self):
        pass

    @abstractmethod
    def subscribe(self):
        pass

    @abstractmethod
    def unsubscribe(self):
        pass

    @abstractmethod
    def _root(self):
        pass


class RuntimeTokenSpace(ABC):
    """The RuntimeTokenSpace abstract base class offers a helpful way to simulate, visualize, and troubleshoot the flow
    of data and algorithms within machine learning models.
    Key Use Cases: Simulating data flow, algorithm planning and development, core data structures, and abstract
    class definitions which can be used to simulate data flow and model visualization/debugging/development.

    Limitations: It might not be the ideal tool for replicating the intricate calculations within hidden layers
    of neural networks or the flow of machine learning models. For tasks like advanced vector retrieval,
    weighting, and biasing, runtime token space interacts with libraries like NumPy, Jax, or TensorFlow."""
    @abstractmethod
    def push(self, item):
        pass

    @abstractmethod
    def pop(self):
        pass

    @abstractmethod
    def peek(self):
        pass


class BaseRuntime(ABC):
    """
    Abstract base class for custom runtimes.
    """

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def __enter__(self):
        pass

    @abstractmethod
    def __exit__(self, exc_type, exc_value, traceback):
        pass

    @abstractmethod
    def _root(self):
        pass
