import logging
import logging.config
from logging.config import dictConfig
import threading
import sys
from pathlib import Path


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
