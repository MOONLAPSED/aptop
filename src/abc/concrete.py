import uuid
import threading
import logging
from openai import OpenAI
from typing import Optional
from abc import ABC
from src.abc.protocol import BaseContextManager, BaseProtocol, RuntimeTokenSpace, BaseRuntime


class OpenAIContextManager(BaseContextManager, ABC, threading.Thread):
    def __init__(self, api_key: str, engine: str, endpoint: str, model: str, **kwargs):
        super().__init__(**kwargs)
        self.api_key = api_key
        self.engine = engine
        self.endpoint = endpoint
        self.model = model
        self.kwargs = kwargs
        self.openai = OpenAI(api_key=self.api_key)

    def __enter__(self):
        return self.openai

    def __exit__(self, exc_type, exc_value, traceback):
        super().__exit__(exc_type, exc_value, traceback)
        pass


class ThreadSafeContextManager(BaseContextManager, ABC):
    """
    Custom thread-safe context manager with lock and UUID.
    """
    def __init__(self):
        super().__init__()
        self.lock = threading.Lock()
        self.uuid = uuid.uuid4()

    def __enter__(self):
        self.lock.acquire()
        print(f"Entering context with UUID: {self.uuid}")
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        try:
            if exc_type is not None:
                print(f"Exception caught in UUID {self.uuid}: {exc_value}")
            # Any other resource cleanup can be done here.
        finally:
            self.lock.release()
            print(f"Exiting context with UUID: {self.uuid}")
            # Optionally, return True to suppress the exception if handled

    @classmethod
    def _root(cls):
        return logging.getLogger(cls.__name__)


class SimpleTokenStack(RuntimeTokenSpace, ABC):
    def __init__(self):
        super().__init__()
        self._stack = []

    def push(self, item):
        self._stack.append(item)

    def pop(self):
        if not self._stack:
            raise IndexError("Stack is empty")
        return self._stack.pop()

    def peek(self):
        if not self._stack:
            raise IndexError("Stack is empty")
        return self._stack[-1]

    def __len__(self):
        return len(self._stack)


class PosixPlayground(BaseRuntime, ABC):
    def __init__(self):
        super().__init__()

    @classmethod
    def _root(cls):
        return logging.getLogger(cls.__name__)

    @classmethod
    def __enter__(cls):
        cls._root().info(f'Entering playground {cls}')
        return cls

    @classmethod
    def __exit__(cls, exc_type, exc_value, traceback):
        for e in [exc_type, exc_value, traceback]:
            if e is not None:
                cls._root().error(f'Exception caught: {e}')
        cls._root().info(f'Exiting playground {cls}')
        pass

    @staticmethod
    def process_file(filepath: str, manager: ThreadSafeContextManager):
        """
        Example of how to use context manager with UUID inside the processing function.
        """
        with manager:
            print(f"Processing file {filepath} with UUID: {manager.uuid}")
            # Load and process the file, make API calls, etc.
            pass


class PosixProtocol(BaseProtocol, ABC):
    def __init__(self):
        super().__init__()

    @classmethod
    def _root(cls):
        return logging.getLogger(cls.__name__)

    @classmethod
    def __enter__(cls):
        cls._root().info(f'Entering protocol {cls}')
        return cls

    @classmethod
    def __exit__(cls, exc_type, exc_value, traceback):
        for e in [exc_type, exc_value, traceback]:
            if e is not None:
                cls._root().error(f'Exception caught: {e}')
        cls._root().info(f'Exiting protocol {cls}')
        pass

    @staticmethod
    def _get_token_space(**kwargs: dict) -> Optional[SimpleTokenStack]:
        """Get SimpleTokenStack instance from kwargs if defined."""

        token_keys = ['token_space', 'SimpleTokenStack', 'Token',
                      'tokens', 'token_stack', 'blob', 'AST', '.py', '.md']

        for key in token_keys:
            if key in kwargs:
                return SimpleTokenStack()

        return None

    def send(self):
        pass

    def get(self):
        pass

    def post(self):
        pass

    def publish(self):
        pass

    def subscribe(self):
        pass


if __name__ == "__main__":
    pass
