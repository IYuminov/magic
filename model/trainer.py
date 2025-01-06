import os
from datetime import datetime

import numpy as np
import torch
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, TensorDataset

from tools.config_reader import config
from tools.logger import logger

logger = logger("TRAINER")  # объект логгера
DEVICE = torch.device(
    # определяем устройство для обучения
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

SAVE_DIR = config["weight"]["path_for_save"]


class Fitter(object):
    """
    Класс для обучения модели.
    Принимает на вход:
    generator - объект генератора Conditional WGAN
    discriminator - объект дискриминатора Conditional WGAN
    batch_size - batch size для DataLoader
    n_epochs - колмчество эпох для обучения
    latent_dim - размерность скрытого пространства
    learning_rate - шаг обучения оптимизатора
    n_critic - ???
    """

    def __init__(
        self,
        generator,
        discriminator,
        batch_size=32,
        n_epochs=10,
        latent_dim=1,
        learning_rate=0.0001,
        n_critic=5,
    ):

        self.generator = generator
        self.discriminator = discriminator
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.latent_dim = latent_dim
        self.lr = learning_rate
        self.n_critic = n_critic

        self.opt_gen = torch.optim.RMSprop(self.generator.parameters(), lr=self.lr)

        self.opt_disc = torch.optim.RMSprop(self.discriminator.parameters(), lr=self.lr)

        self.generator.to(DEVICE)
        self.discriminator.to(DEVICE)

    def fit(self, X, y):
        """
        Функция для обучения модели
        """

        try:
            # numpy to tensor
            X_real = torch.tensor(X, dtype=torch.float, device=DEVICE)

            y_cond = torch.tensor(y, dtype=torch.float, device=DEVICE)

            dataset_real = TensorDataset(X_real, y_cond)  # tensor to dataset

            logger.info("Data prepared.")

        except Exception as e:
            logger.error(f"Data preparation error: {e}")

        # Turn on training
        self.generator.train(True)
        self.discriminator.train(True)

        self.loss_history = []

        logger.info("Start of training.")

        # Fit GAN
        for epoch in range(self.n_epochs):
            for i, (real_batch, cond_batch) in enumerate(
                DataLoader(dataset_real, batch_size=self.batch_size, shuffle=True)
            ):

                real_batch, cond_batch = real_batch.to(DEVICE), cond_batch.to(DEVICE)
                self.opt_disc.zero_grad()
                noise = torch.normal(
                    0, 1, (real_batch.size(0), self.latent_dim), device=DEVICE
                )

                gen_batch = self.generator(noise, cond_batch).detach()
                loss = -torch.mean(
                    self.discriminator(real_batch, cond_batch)
                ) + torch.mean(self.discriminator(gen_batch, cond_batch))
                loss.backward()
                self.opt_disc.step()

                for p in self.discriminator.parameters():
                    p.data.clamp_(-0.01, 0.01)

                if i % self.n_critic == 0:
                    self.opt_gen.zero_grad()
                    noise = torch.normal(
                        0, 1, (real_batch.size(0), self.latent_dim), device=DEVICE
                    )

                    gen_batch = self.generator(noise, cond_batch)
                    loss = -torch.mean(self.discriminator(gen_batch, cond_batch))
                    loss.backward()
                    self.opt_gen.step()

            # caiculate and store loss after an epoch
            Z_noise = torch.normal(0, 1, (len(X_real), self.latent_dim)).to(DEVICE)
            X_fake = self.generator(Z_noise, y_cond)

            loss_epoch = torch.mean(self.discriminator(X_real, y_cond)) - torch.mean(
                self.discriminator(X_fake, y_cond)
            )

            self.loss_history.append(loss_epoch.detach().cpu())

            try:
                if epoch % 10 == 0:

                    """
                    Сохранение весов модели каждые 10 эпох
                    """

                    model_file_name = datetime.now().strftime(
                        f"model_%Y_%m_%d_%H:%M__{epoch}_epoch.pth"
                    )
                    path = os.path.join(SAVE_DIR, model_file_name)
                    torch.save(self.generator.cpu().state_dict(), path)
                    self.generator.to(DEVICE)
                    logger.info(f"File saved: {model_file_name}")

            except Exception as e:
                logger.error(f"Error saving model on {epoch} epoch: {e}")

        # Turn off training
        self.generator.train(False)
        self.discriminator.train(False)
        logger.info("Training completed.")

    def generate(self, target, latent_dim):
        """
        Функция для гененрации параметров кластеров
        Вызывается для скоринга в функции scoring
        """

        try:
            target_target = torch.tensor(target, dtype=torch.float, device=DEVICE)
            noise = torch.normal(
                0, 1, (target_target.size(0), latent_dim), device=DEVICE
            )
            generate_data = self.generator(noise, target_target).detach().cpu()
            logger.info("Data successfully generated.")

        except Exception as e:
            logger.error(f"Error generating data: {e}")

        return generate_data.numpy()

    def scoring(self, data_train, data_test, target_train, target_test, latent_dim):
        """
        Функция для оценки качества генерации по имеющимся данным
        """

        try:
            generated_data_train = self.generate(target_train, latent_dim)
            generated_data_test = self.generate(target_test, latent_dim)

            # собираем реальный и фейковые матрицы в одну
            concat_data_train = np.concatenate(
                (generated_data_train, data_train), axis=0
            )
            concat_data_test = np.concatenate((generated_data_test, data_test), axis=0)

            generate_target_train = np.array(
                [0] * len(generated_data_train) + [1] * len(data_train)
            )
            generate_target_test = np.array(
                [0] * len(generated_data_test) + [1] * len(data_test)
            )

            estimator = (
                GradientBoostingClassifier()
            )  # оценщик для скоринга результата генерации
            estimator.fit(concat_data_train, generate_target_train)
            estimator_prediction = estimator.predict_proba(concat_data_test)[
                :, 1
            ]  # получаем генерацию данных

            metric_value = roc_auc_score(generate_target_test, estimator_prediction)
            logger.info(f"Metric of generate data: {metric_value}")

        except Exception as e:
            logger.error(f"Error scoring generated data: {e}")
