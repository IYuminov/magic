from pickle import load

from boto3 import Session

from tools.config_reader import config
from tools.logger import logger

logger = logger("DATA_READER")


def data_reader_func(file_name: str):
    """
    Функция cчитывает данные для обучения из S3 хранилища в директорию среду исполнения.
    После этого, данные из директории передаются в переменную и возвращаются функцией.
    Принимает на вход имя файла.
    """

    session = Session(
        aws_access_key_id=config["s3"]["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=config["s3"]["AWS_SECRET_ACCESS_KEY"],
    )

    s3_client = session.client("s3", endpoint_url=config["s3"]["ENDPOINT"])

    # Скачиваем файл из S3
    with open(f"{config['data']['path_for_save']}{file_name}", "wb") as file:
        s3_client.download_fileobj(
            Bucket=config["s3"]["BUCKET_NAME"],
            Key=f"{config['data']['path_for_dowload']}{file_name}",
            Fileobj=file,
        )
    logger.info(
        f"File {file_name} saved to {config['data']['path_for_save']} directory from S3."
    )

    with open(f"{config['data']['path_for_save']}{file_name}", "rb") as data:
        data = load(data)

    return data
