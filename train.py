from model.discriminator import Discriminator
from model.generator import Generator
from model.trainer import Fitter
from tools.config_reader import config
from tools.data_reader import data_reader_func

# Читаем данные для обучения
X_train, y_train = data_reader_func(config["data"]["data_train"]), data_reader_func(
    config["data"]["target_train"]
)
X_test, y_test = data_reader_func(config["data"]["data_test"]), data_reader_func(
    config["data"]["target_test"]
)

# Объект генератора Conditional WGAN
generator = Generator(
    config["model"]["latent_dim"] + config["data"]["n_inputs"],  # n_inputs
    config["data"]["n_outputs"],  # n_outputs
)


# Объект дискриминатора Conditional WGAN
discriminator = Discriminator(
    config["data"]["n_outputs"] + config["data"]["n_inputs"]  # n_inputs
)

# Объект класса обучения Conditional WGAN
fitter = Fitter(
    generator,
    discriminator,
    batch_size=config["model"]["batch_size"],
    n_epochs=config["model"]["n_epochs"],
    latent_dim=config["model"]["latent_dim"],
    learning_rate=config["model"]["learning_rate"],
    n_critic=config["model"]["n_critic"],
)

if __name__ == "__main__":
    fitter.fit(X_train, y_train)
    fitter.scoring(X_train, X_test, y_train, y_test, config["model"]["latent_dim"])
