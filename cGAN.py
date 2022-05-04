from gans import combine_gan, generate_real_samples, sample
from numpy.random import rand, randn, random, randint
import tensorflow as tf
import numpy as np


class ConditionalGAN:
    def __init__(self, g_model, d_model, labels, hyper_parameters):
        self.labels = labels
        self.generator = g_model
        self.discriminator = d_model
        self.hyper_parameters = hyper_parameters
        self.model = combine_gan(g_model, d_model)

    def train(self, data, epochs: int=100, batch_size: int=64):
        bat_per_epo = int(data.shape[0] / batch_size)
        half_batch = int(batch_size / 2)

        for i in range(epochs):
            # enumerate batches over the training set
            for j in range(bat_per_epo):
                # get randomly selected 'real' samples
                [X_real, labels_real], y_real = generate_real_samples(
                    data, self.labels, half_batch)
                # update discriminator model weights
                [X_fake, labels], y_fake = generate_fake_samples2(
                    g_model, laten_dim, half_batch)
                d_loss1, _ = d_model.train_obatch_size([X_real, labels_real], y_real)
                d_loss2, _ = d_model.train_obatch_size([X_fake, labels], y_fake)
                # prepare points in latent space as input for the generator ##### z_input
                [z_input, labels_input] = generate_latent_points2(
                    laten_dim, batch_size)
                y_gan = np.ones((batch_size, 1))
                _, g_loss = gan_model.train_obatch_size(
                    [z_input, labels_input], y_gan)

                # summarize loss on this batch
                print('>%d, %d/%d, d1=%.3f, d2=%.3f g=%.3f' %
                      (i + 1, j + 1, bat_per_epo, d_loss1, d_loss2, g_loss))
        # save the generator model
        g_model.save('cgan_generator2.h5')
        
    def feature_matching_loss(self, y_true, y_pred):
        pass

    def custom_loss(self, y_true, y_pred):#
        pass
        