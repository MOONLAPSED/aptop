import logging
import logging.config
from logging.config import dictConfig
import threading
import sys
from pathlib import Path
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Generic, List, TypeVar, Union
import struct
import functools

T = TypeVar('T')

class Atom(ABC):
    @abstractmethod
    def encode(self) -> bytes:
        pass

    @abstractmethod
    def decode(self, data: bytes) -> None:
        pass

    @abstractmethod
    def execute(self, *args, **kwargs) -> Any:
        pass

    @abstractmethod
    def __repr__(self):
        pass

    @abstractmethod
    def to_dataclass(self):
        pass


@dataclass(frozen=True)
class AtomDataclass(Generic[T], Atom):
    value: T
    data_type: str = field(init=False)

    def __post_init__(self):
        data_type_name = type(self.value).__name__
        object.__setattr__(self, 'data_type', data_type_name.lower())

    def __repr__(self):
        return f"AtomDataclass(id={id(self)}, value={self.value}, data_type='{self.data_type}')"

    def to_dataclass(self):
        return self

    def __add__(self, other):
        return AtomDataclass(self.value + other.value)

    def __sub__(self, other):
        return AtomDataclass(self.value - other.value)

    def __mul__(self, other):
        return AtomDataclass(self.value * other.value)

    def __truediv__(self, other):
        return AtomDataclass(self.value / other.value)

    def __eq__(self, other):
        if isinstance(other, AtomDataclass):
            return self.value == other.value
        return False

    def __lt__(self, other):
        if isinstance(other, AtomDataclass):
            return self.value < other.value
        return False

    def __le__(self, other):
        if isinstance(other, AtomDataclass):
            return self.value <= other.value
        return False

    def __gt__(self, other):
        if isinstance(other, AtomDataclass):
            return self.value > other.value
        return False

    def __ge__(self, other):
        if isinstance(other, AtomDataclass):
            return self.value >= other.value
        return False

    @staticmethod
    def _determine_data_type(data: Any) -> str:
        data_type = type(data).__name__
        if data_type == 'str':
            return 'string'
        elif data_type == 'int':
            return 'integer'
        elif data_type == 'float':
            return 'float'
        elif data_type == 'bool':
            return 'boolean'
        elif data_type == 'list':
            return 'list'
        elif data_type == 'dict':
            return 'dictionary'
        else:
            return 'unknown'

    def encode(self) -> bytes:
        data_type_bytes = self.data_type.encode('utf-8')
        data_bytes = self._encode_data()
        header = struct.pack('!I', len(data_bytes))
        return header + data_type_bytes + data_bytes
    
    def _encode_data(self) -> bytes:
        if self.data_type == 'string':
            return self.value.encode('utf-8')
        elif self.data_type == 'integer':
            return struct.pack('!q', self.value)
        elif self.data_type == 'float':
            return struct.pack('!d', self.value)
        elif self.data_type == 'boolean':
            return struct.pack('?', self.value)
        elif self.data_type == 'list':
            return b''.join([AtomDataclass(element)._encode_data() for element in self.value])
        elif self.data_type == 'dictionary':
            return b''.join(
                [AtomDataclass(key)._encode_data() + AtomDataclass(value)._encode_data() for key, value in self.value.items()]
            )
        else:
            raise ValueError(f"Unsupported data type: {self.data_type}")
    
    def decode(self, data: bytes) -> None:
        header_length = struct.unpack('!I', data[:4])[0]
        data_type_bytes = data[4:4 + header_length]
        data_type = data_type_bytes.decode('utf-8')
        data_bytes = data[4 + header_length:]

        if data_type == 'string':
            value = data_bytes.decode('utf-8')
        elif data_type == 'integer':
            value = struct.unpack('!q', data_bytes)[0]
        elif data_type == 'float':
            value = struct.unpack('!d', data_bytes)[0]
        elif data_type == 'boolean':
            value = struct.unpack('?', data_bytes)[0]
        elif data_type == 'list':
            value = []
            offset = 0
            while offset < len(data_bytes):
                element = AtomDataclass(None)
                element_size = element.decode(data_bytes[offset:])
                value.append(element.value)
                offset += element_size
        elif data_type == 'dictionary':
            value = {}
            offset = 0
            while offset < len(data_bytes):
                key = AtomDataclass(None)
                key_size = key.decode(data_bytes[offset:])
                offset += key_size
                val = AtomDataclass(None)
                value_size = val.decode(data_bytes[offset:])
                offset += value_size
                value[key.value] = val.value
        else:
            raise ValueError(f"Unsupported data type: {data_type}")

        self.value = value
        object.__setattr__(self, 'data_type', data_type)

    def execute(self, *args, **kwargs) -> Any:
        pass


@dataclass
class FormalTheory(Generic[T]):
    reflexivity: Callable[[T], bool] = lambda x: x == x
    symmetry: Callable[[T, T], bool] = lambda x, y: x == y
    transitivity: Callable[[T, T, T], bool] = lambda x, y, z: (x == y) and (y == z) and (x == z)
    transparency: Callable[[Callable[..., T], T, T], T] = lambda f, x, y: f(True, x, y) if x == y else None
    case_base: Dict[str, Callable[[T, T], T]] = field(default_factory=dict)
    
    def __post_init__(self):
        self.case_base = {
            '⊤': lambda x, _: x,
            '⊥': lambda _, y: y,
            'a': self.if_else_a
        }
    
    @staticmethod
    def if_else(a: bool, x: T, y: T) -> T:
        return x if a else y

    def if_else_a(self, x: T, y: T) -> T:
        return self.if_else(True, x, y)

    def compare(self, atoms: List[AtomDataclass[T]]) -> bool:
        if not atoms:
            return False
        comparison = [self.symmetry(atoms[0].value, atoms[i].value) for i in range(1, len(atoms))]
        return all(comparison)

    def __repr__(self):
        case_base_repr = {
            key: (value.__name__ if callable(value) else value)
            for key, value in self.case_base.items()
        }
        return (f"FormalTheory(\n"
                f"  reflexivity={self.reflexivity.__name__},\n"
                f"  symmetry={self.symmetry.__name__},\n"
                f"  transitivity={self.transitivity.__name__},\n"
                f"  transparency={self.transparency.__name__},\n"
                f"  case_base={case_base_repr}\n"
                f")")


_lock = threading.Lock()


def _init_basic_logging():
    basic_log_file_path = Path(__file__).resolve().parent.joinpath('logs', 'setup.log')
    basic_log_file_path.parent.mkdir(parents=True, exist_ok=True)  # Ensure the logs directory exists
    logging.basicConfig(
        filename=str(basic_log_file_path),  # Convert Path object to string for compatibility
        level=logging.INFO,
        format='[%(levelname)s]%(asctime)s||%(name)s: %(message)s',
        datefmt='%Y-%m-%d~%H:%M:%S%z',
    )


# Initialize basic logging immediately to capture any issues during module import.
_init_basic_logging()


def main() -> logging.Logger:
    """Configures logging for the app.

    Returns:
        logging.Logger: The logger for the module.
    """
    # Find the current directory for logging
    current_dir = Path(__file__).resolve().parent
    while not (current_dir / 'logs').exists():
        current_dir = current_dir.parent
        if current_dir == Path('/'):
            break
    # Ensure the logs directory exists
    logs_dir = Path(__file__).resolve().parent.joinpath('logs')
    logs_dir.mkdir(exist_ok=True)
    # Add paths for importing modules
    sys.path.append(str(Path(__file__).resolve().parent))
    sys.path.append(str(Path(__file__).resolve().parent.joinpath('src')))
    with _lock:
        logging_config = {
            'version': 1,
            'disable_existing_loggers': False,
            'formatters': {
                'default': {
                    'format': '[%(levelname)s]%(asctime)s||%(name)s: %(message)s',
                    'datefmt': '%Y-%m-%d~%H:%M:%S%z'
                },
            },
            'handlers': {
                'console': {
                    'level': 'INFO',  # Explicitly set level to 'INFO'
                    'class': 'logging.StreamHandler',
                    'formatter': 'default',
                    'stream': 'ext://sys.stdout'
                },
                'file': {
                    'level': 'INFO',  # Explicitly set level to 'INFO'
                    'formatter': 'default',
                    'class': 'logging.handlers.RotatingFileHandler',
                    'filename': str(logs_dir / 'app.log'),  # Convert Path object to string for compatibility
                    'maxBytes': 10485760,  # 10MB
                    'backupCount': 10
                }
            },
            'root': {
                'level': logging.INFO,
                'handlers': ['console', 'file']
            }
        }

        dictConfig(logging_config)

        logger = logging.getLogger(__name__)
        logger.info(f'\nSource_file: {__file__}|'
                    f'\nWorking_dir: {current_dir}|')

        return logger


if __name__ == '__main__':
    main()
    # Example AtomDataclass usage:
    atom1 = AtomDataclass(5)
    atom2 = AtomDataclass(5)
    # Example FormalTheory usage:
    formal_theory = FormalTheory[int]()

    print(atom1)              # AtomDataclass(id=..., value=5, data_type='int')
    print(atom2)              # AtomDataclass(id=..., value=5, data_type='int')
    print(formal_theory.compare([atom1, atom2]))  # True
    print(formal_theory)      # Display representation of FormalTheory instance