import logging
import sys


def logger(module_name: str):
    """
    Настраивает логгер для записи событий в файл и вывода в stdout.
    """
    # Создаем логгер
    logger = logging.getLogger(module_name)
    logger.setLevel(logging.DEBUG)

    # # Создаем обработчик для записи в файл
    # file_handler = logging.FileHandler("events.log")
    # file_handler.setLevel(logging.DEBUG)

    # Создаем обработчик для вывода в stdout
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(logging.DEBUG)

    # Настраиваем формат логов
    formatter = logging.Formatter(
        f"%(asctime)s - {module_name} - %(levelname)s - %(message)s"
    )
    # file_handler.setFormatter(formatter)
    stdout_handler.setFormatter(formatter)

    # Добавляем обработчики к логгеру
    # logger.addHandler(file_handler)
    logger.addHandler(stdout_handler)

    return logger
