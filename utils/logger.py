"""Centralised logging via loguru — single import for all modules."""
import sys
from pathlib import Path
from loguru import logger

LOG_DIR = Path(__file__).parent.parent / "logs"
LOG_DIR.mkdir(exist_ok=True)


def setup_logger(level: str = "INFO") -> None:
    logger.remove()
    logger.add(sys.stdout, level=level, colorize=True,
               format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | "
                      "<cyan>{name}</cyan>:<cyan>{line}</cyan> — <level>{message}</level>")
    logger.add(LOG_DIR / "algotrad_{time:YYYY-MM-DD}.log",
               level=level, rotation="1 day", retention="30 days", compression="zip")


setup_logger()

__all__ = ["logger"]
