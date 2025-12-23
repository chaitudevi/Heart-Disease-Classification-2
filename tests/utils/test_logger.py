import logging
import pytest

from src.utils.logger import get_logger

def test_get_logger_returns_logger_instance():
    """
    Ensure get_logger returns a logging.Logger object.
    """
    logger = get_logger("test_logger")
    assert isinstance(logger, logging.Logger)

def test_get_logger_adds_handler_once():
    """
    Ensure handler is added only once even if get_logger is called multiple times.
    """
    logger = get_logger("test_logger_once")
    handlers_before = len(logger.handlers)

    logger2 = get_logger("test_logger_once")
    handlers_after = len(logger2.handlers)

    assert handlers_before == handlers_after
    assert logger is logger2

def test_logger_formatter_and_level():
    """
    Ensure logger has INFO level and the correct formatter string.
    """
    logger = get_logger("test_logger_format")
    assert logger.level == logging.INFO

    handler = logger.handlers[0]
    formatter = handler.formatter
    assert formatter._fmt == "[%(asctime)s] %(levelname)s | %(name)s | %(message)s"

def test_logger_emits_message(caplog):
    """
    Ensure logger actually emits a message with the expected format.
    """
    logger = get_logger("test_logger_emit")
    with caplog.at_level(logging.INFO):
        logger.info("Hello World")

    # Check captured log
    assert "Hello World" in caplog.text
    assert "test_logger_emit" in caplog.text
    assert "INFO" in caplog.text
