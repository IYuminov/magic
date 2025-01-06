import os

import yaml
from dotenv import load_dotenv

from tools.logger import logger

logger = logger("CONFIG_READER")


def load_config_with_env(config_path, env_path=".env"):
    """
    Загружает YAML конфиг и подставляет значения из .env-файла в места с шаблонами вида ${VAR_NAME}.

    Args:
        config_path (str): Путь к конфиг-файлу (YAML).
        env_path (str): Путь к .env-файлу (по умолчанию ".env").

    Returns:
        dict: Конфиг с подставленными переменными окружения.
    """
    # Загружаем переменные из .env
    load_dotenv(dotenv_path=env_path)

    # Читаем YAML конфиг
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    # Рекурсивная функция для замены шаблонов в конфиге
    def replace_env_variables(obj):
        if isinstance(obj, dict):
            return {key: replace_env_variables(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [replace_env_variables(item) for item in obj]
        elif isinstance(obj, str) and obj.startswith("${") and obj.endswith("}"):
            # Извлекаем имя переменной окружения и подставляем значение
            env_var = obj[2:-1]
            return os.getenv(env_var, obj)
        else:
            return obj

    logger.info("Config readed.")

    # Заменяем переменные в конфиге
    return replace_env_variables(config)


config = load_config_with_env("configs/config.yaml", ".env")
