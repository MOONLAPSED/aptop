import uuid
import threading
import logging
from openai import OpenAI
from abc import ABC
from src.app.protocol import BaseContextManager


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


class OpenAIContext(ThreadSafeContextManager, OpenAI, ABC):
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

    client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")

    history = [
        {"role": "system",
         "content": "You are an intelligent assistant. Provide well-reasoned answers that are both correct and helpful."},
        {"role": "user",
         "content": "Hello, introduce yourself to someone opening this program for the first time. Be concise."},
    ]

    while True:
        completion = client.chat.completions.create(
            messages=history,
            temperature=0.7,
            stream=True,
        )

        new_message = {"role": "assistant", "content": ""}

        for chunk in completion:
            if chunk.choices[0].delta.content:
                print(chunk.choices[0].delta.content, end="", flush=True)
                new_message["content"] += chunk.choices[0].delta.content

        history.append(new_message)

        print()
        history.append({"role": "user", "content": input("> ")})