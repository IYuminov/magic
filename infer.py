from datetime import datetime

import torch
from boto3 import Session

from model.generator import Generator
from tools.config_reader import config
from tools.data_reader import data_reader_func
from tools.finder_file import find_latest_file
from tools.logger import logger

logger = logger("INFER")


PATH = find_latest_file(
    config["weight"]["path_for_save"]
)  # находим актуальный вес для инициализации модели

# Таргет, по которому необходимо сгенерировать  параметры кластеров
target_for_generate = data_reader_func(config["data"]["target_test"])

# Объект генератора Conditional WGAN
generator = Generator(
    config["model"]["latent_dim"] + config["data"]["n_inputs"],  # n_inputs
    config["data"]["n_outputs"],  # n_outputs
)
generator.load_state_dict(torch.load(PATH, weights_only=True))
generator.eval()


def main(gen_machine, target, latent_dim):
    """
    Функция для генерации параметров кластера по заданному таргету.

    Принимает:
    gen_machine - модель для генерации параметров кластера
    target - таргет по которому будут сгенерированы параметры кластера
    """

    target_target = torch.tensor(target, dtype=torch.float)
    noise = torch.normal(0, 1, (target_target.size(0), latent_dim))
    generated_data = gen_machine(noise, target_target).detach().cpu()
    model_file_name = datetime.now().strftime("generated_data_%Y_%m_%d_%H:%M.pt")
    torch.save(generated_data, f"{config['data']['result_save_path']}{model_file_name}")

    logger.info(
        f"""Data successfully generated and saved.
        Path: {config['data']['result_save_path']}{model_file_name}
        Shape: {generated_data.shape}"""
    )

    try:
        session = Session(
            aws_access_key_id=config["s3"]["AWS_ACCESS_KEY_ID"],
            aws_secret_access_key=config["s3"]["AWS_SECRET_ACCESS_KEY"],
        )

        s3_client = session.client("s3", endpoint_url=config["s3"]["ENDPOINT"])

        path_upload_file_local = (
            config["data"]["result_save_path"] + model_file_name
        )  # путь на локальном диске до файла для загрузки в S3

        local = config["data"]["path_for_dowload"]
        s3 = config["data"]["result_save_path"]

        path_upload_file_s3 = local + s3 + model_file_name
        # путь до места назначения файла на S3

        s3_client.upload_file(
            path_upload_file_local, config["s3"]["BUCKET_NAME"], path_upload_file_s3
        )

        logger.info(
            f"""Result uploaded in S3 bucket.
            Path: {path_upload_file_s3}"""
        )

    except Exception as e:
        logger.error(f"Error uploading result in S3: {e}")


if __name__ == "__main__":
    main(generator, target_for_generate, config["model"]["latent_dim"])
