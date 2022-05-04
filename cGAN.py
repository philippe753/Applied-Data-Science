import keras.losses

from gans import combine_gan, generate_real_samples, generate_fake_samples, generate_latent_points, keras, generator, discriminator
from keras import backend, Model
import pandas as pd
import numpy as np

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class HyperParameters:
    def __init__(self, latent_dim: int=50, learning_rate=1e-5):
        self.latent_dim = latent_dim
        self.learning_rate = learning_rate


class ConditionalGAN:
    def __init__(self, g_model, d_model, labels, hyper_parameters: HyperParameters()):
        self.labels = labels
        self.generator = g_model
        self.discriminator = d_model

        self.hyper_parameters = hyper_parameters

        opt = keras.optimizers.Adam(lr=hyper_parameters.learning_rate)
        self.generator.compile(loss="binary_crossentropy", optimizer=opt, metrics='accuracy')

        self.model = combine_gan(g_model, d_model)
        self.model.trainable = True

    def train(self, data, epochs: int=100, batch_size: int=64):
        bat_per_epo = int(data.shape[0] / batch_size)
        half_batch = int(batch_size / 2)

        for i in range(epochs):
            for j in range(bat_per_epo):
                d_loss1, d_loss2, g_loss = self.train_mini_batch(data, batch_size, half_batch)

                if j % 100 == 0:
                    self.print_loss(i, j, bat_per_epo, d_loss1, d_loss2, g_loss)
        self.save()

    def train_mini_batch(self, data, batch_size: int, half_batch_size: int) -> (float, float, float):
        [X_real, labels_real], y_real = generate_real_samples(data, self.labels, half_batch_size)
        [X_fake, labels], y_fake = generate_fake_samples(self.generator, self.hyper_parameters.latent_dim, half_batch_size)
        d_loss1, _ = self.discriminator.train_on_batch([X_real, labels_real], y_real)
        d_loss2, _ = self.discriminator.train_on_batch([X_fake, labels], y_fake)

        # Prepare points in latent space as input for the generator
        [z_input, labels_input] = generate_latent_points(self.hyper_parameters.latent_dim, batch_size)
        y_gan = np.ones((batch_size, 1))
        _, g_loss = self.generator.train_on_batch([z_input, labels_input], y_gan)

        return d_loss1, d_loss2, g_loss

    def save(self) -> None:
        self.generator.save('cgan_generator2.h5')
        self.discriminator.save('cgan_discriminator2.h5')

    def feature_matching_loss(self, y_true, y_pred):
        """ Binary Cross Entropy Feature Matching Loss """
        intermediate_layer_model = self.intermediate_layer()

        f_x_given_y = intermediate_layer_model(y_pred, y_true)
        f_g_of_z = intermediate_layer_model(y_pred)

        f_x_given_y = backend.cast_to_floatx(f_x_given_y)
        f_g_of_z = backend.cast_to_floatx(f_g_of_z)

        loss = backend.sqrt(backend.square(f_x_given_y - f_g_of_z))
        loss = backend.sum(loss)
        print(f"FM Loss: {loss}")
        return loss

    def custom_loss(self, y_true, y_pred):
        fm_loss = self.feature_matching_loss(y_true, y_pred)
        bce_d_loss = keras.losses.bce(y_true, y_pred)

        final_loss = fm_loss + bce_d_loss
        return final_loss

    def intermediate_layer(self):
        layer_name = 'intermediate_layer'

        keras.utils.plot_model(self.discriminator, to_file="model3.png", show_shapes=True, show_dtype=True)
        intermediate_layer_model = Model(inputs=self.discriminator.input,
                                         outputs=self.discriminator.output)
        return intermediate_layer_model

    @staticmethod
    def print_loss(i, j, bat_per_epo, d_loss1, d_loss2, g_loss):
        print('>%d, %d/%d, d1=%.3f, d2=%.3f g=%.3f' %
              (i + 1, j + 1, bat_per_epo, d_loss1, d_loss2, g_loss))


def main():
    hyper_params = HyperParameters(latent_dim=50)

    df = pd.read_csv(r"Q:\MichaelsStuff\EngMaths\Year4\AppliedDataScience\creditcard.csv")

    columns_ = df.columns[2:30]
    dataset = df[columns_]  # X
    labels = df.Class  # y.Class

    d_model = discriminator()
    # create the generator
    g_model = generator(hyper_params.latent_dim)

    philippe = ConditionalGAN(g_model, d_model, labels=labels, hyper_parameters=hyper_params)
    philippe.train(dataset)


if __name__ == "__main__":
    main()
