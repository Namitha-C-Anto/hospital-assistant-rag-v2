import logging
import sys

def configure_logger() -> logging.Logger:
    """
    Configure and return the application logger.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
        ],
    )

    logger = logging.getLogger("HospitalAssistant")

    return logger

logger = configure_logger()