from __future__ import annotations
import keras.losses
import tensorflow

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

        self.model = combine_gan(self.generator, d_model)

        self.opt = keras.optimizers.Adam(lr=hyper_parameters.learning_rate)

        self.model.compile(optimizer=self.opt, metrics='accuracy')
        self.model.trainable = True

        keras.utils.plot_model(self.model, to_file="model3.png", show_shapes=True, show_dtype=True)

    def train(self, data, epochs: int = 100, batch_size: int = 64):
        bat_per_epo = int(data.shape[0] / batch_size)
        half_batch = int(batch_size / 2)

        for i in range(epochs):
            for j in range(bat_per_epo):
                final_loss, d_loss1, d_loss2 = self.train_mini_batch(data, batch_size, half_batch)

                if j % 100 == 0:
                    self.print_loss(i, j, bat_per_epo, d_loss1, d_loss2, final_loss)
            self.save()

    def train_mini_batch(self, data, batch_size: int, half_batch_size: int) -> (float, float, float):
        # Prepare points in latent space as input for the generator
        # [z_input, labels_input] = generate_latent_points(self.hyper_parameters.latent_dim, batch_size)
        # y_gan = np.ones((batch_size, 1))

        with tensorflow.GradientTape() as tape:
            [X_real, labels_real], y_real = generate_real_samples(data, self.labels, half_batch_size)
            [X_fake, labels], y_fake = generate_fake_samples(self.generator, self.hyper_parameters.latent_dim,
                                                             half_batch_size)
            pred_1, feature_vector1 = self.discriminator([X_real, backend.constant(labels_real)])
            pred_2, feature_vector2 = self.discriminator([X_fake, backend.constant(labels)])

            fm_loss = keras.losses.MeanSquaredError()(feature_vector1, feature_vector2)

            d_loss1 = keras.losses.BinaryCrossentropy(from_logits=True)(pred_1,  y_real)
            d_loss2 = keras.losses.BinaryCrossentropy(from_logits=True)(pred_2,  y_fake)

            fm_loss = tensorflow.cast(fm_loss, dtype=tensorflow.float32)
            d_loss1 = tensorflow.cast(d_loss1, dtype=tensorflow.float32)
            d_loss2 = tensorflow.cast(d_loss2, dtype=tensorflow.float32)

            final_loss = fm_loss + (d_loss1 + d_loss2) / 2
            # print("Trainable weights", self.model.trainable_weights)
            gradients = tape.gradient(final_loss, self.model.trainable_weights)
            self.opt.apply_gradients(zip(gradients, self.model.trainable_weights))

        return final_loss, d_loss1, d_loss2

    def save(self) -> None:
        self.generator.save('cgan_generator2.h5')
        self.discriminator.save('cgan_discriminator2.h5')

    def load(self, path: str) -> ConditionalGAN:
        self.model.load_weights(path)


    def feature_matching_loss(self, data, z):
        """ Binary Cross Entropy Feature Matching Loss """

        left_bit = self.discriminator(data).get_layer("intermediate_layer").output
        right_bit = self.discriminator(self.generator(z)).get_layer("intermediate_layer").output

        left_bit = backend.cast_to_floatx(left_bit)
        right_bit = backend.cast_to_floatx(right_bit)

        loss = backend.sqrt(backend.square(left_bit - right_bit))
        loss = backend.sum(loss)
        print(f"FM Loss: {loss}")
        return loss

    @staticmethod
    def print_loss(i, j, bat_per_epo, d_loss1, d_loss2, final_loss):
        print('>%d, %d/%d, d1=%.3f, d2=%.3f T=%.3f' %
              (i + 1, j + 1, bat_per_epo, d_loss1, d_loss2, final_loss))


def main():
    hyper_params = HyperParameters(latent_dim=50, learning_rate=0.000_1)

    df = pd.read_csv(r"C:\Users\Mike Nelhams\OneDrive\Documents\MikesStuff2\Coding\AppliedDataScience\PROJECT"
                     r"\Applied-Data-Science\Datasets\creditcard.csv")

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
