import math
from turtle import shape
from sklearn.linear_model import LinearRegression
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from numpy.random import normal, randint
from tensorflow.keras import layers
from keras.models import load_model, Model
import numpy as np
from numpy.random import rand, randn, random
from numpy.random import randint
import matplotlib.pyplot as plt
import keras.backend as K
from keras.utils.vis_utils import plot_model

sample = ((tf.Tensor, np.array), np.array)

# Whole dataFrame
df = pd.read_csv('creditcard.csv')

# # Import whole dataset
# df_balanced = pd.read_csv('balanced_data.csv')

# # Import balanced dataset
# df_balanced = pd.read_csv('balanced_data.csv')
# # Pick only the froud data
# froud = df_balanced[df_balanced["Class"] == 1]
# # pd datatrame of only the Vs
# froud_V_colums = froud.columns[2:30]
# print(froud_V_colums)
# df_froud_Vs = froud[froud_V_colums]
#####################
df_10_per = df.sample(n=math.floor(len(df)/10))
X, y = df_10_per.iloc[:, 1:-2], df_10_per.iloc[:, -1:]

# Select real samples


def generate_real_samples2(dataset, labels, n_samples):
    # split into images and labels
    # choose random instances
    idx = randint(0, dataset.shape[0], n_samples)
    # select images and labels\
    # print('dataset type: ')
    # print(type(dataset))
    # print('labels: ')
    # print(labels)
    X = dataset.iloc[idx]
    if len(labels) == len(dataset):
        labels = labels[idx]
    else:
        labels = labels
    X = tf.reshape(X, [n_samples, 28])
    # print("X", np.shape(X))
    # generate class labels
    y = np.ones((n_samples, 1))
    return [X, labels], y


def generate_fake_samples2(generator, latent_dim, n_samples):
    z_input, labels_input = generate_latent_points2(latent_dim, n_samples)
    # z_input = np.reshape(z_input, (32, -1))
    # print('z_input-------------')
    # print(z_input)

    datapoints = generator.predict([z_input, labels_input])
    # create class labels
    datapoints = tf.reshape(datapoints, [n_samples, 28])
    y = np.zeros((n_samples, 1))
    return [datapoints, labels_input], y


def get_intermediate_layer(d_model):
    # model = discriminator_model()
    layer_name = 'intermediate_layer'
    intermediate_layer_model = Model(inputs=d_model.input,
                                     outputs=d_model.get_layer(layer_name).output)
    return intermediate_layer_model


def feature_matching(d_model, g_model):

    get_intermediate_layer_output = get_intermediate_layer()


def discriminator_model():
    model = keras.Sequential(
        [
            layers.Dense(128, input_shape=(28,), activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(32, activation='relu', name='intermediate_layer'),
            layers.Dropout(0.2),
            layers.Dense(1, activation='sigmoid')
        ]
    )
    # Compile model
    opt = keras.optimizers.Adam(lr=1e-5)
    model.compile(loss='binary_crossentropy',
                  optimizer=opt, metrics='accuracy')
    return model

# define the standalone discriminator model


def define_discriminator(n_classes=2):  # in_shape=(1,28)

    model = keras.Sequential(
        [
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(32, activation='relu', name='intermediate_layer'),
            layers.Dense(1, activation='sigmoid')

        ]
    )

    gen_sample = layers.Input(shape=(28))
    label = layers.Input(shape=(1,), dtype='int32')
    label_embedding = layers.Flatten()(
        layers.Embedding(n_classes, 28)(label))

    model_input = layers.multiply([gen_sample, label_embedding])
    validity = model(model_input)
    model_dis = Model(inputs=[gen_sample, label],
                      outputs=validity, name="Discriminator")
    opt = keras.optimizers.Adam(lr=1e-5)
    model_dis.compile(loss='binary_crossentropy',
                      optimizer=opt, metrics='accuracy')

    return model_dis


def define_generator(laten_dim, n_classes=2):

    model = keras.Sequential(
        [
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(64, activation='relu', name='intermediate_layer'),
            layers.Dropout(0.2),
            layers.Dense(28, activation='tanh')
        ]
    )

    noise = layers.Input(shape=(laten_dim,))
    label = layers.Input(shape=(1,), dtype='int32')
    label_embedding = layers.Flatten()(layers.Embedding(n_classes, laten_dim)(label))
    model_input = layers.multiply([noise, label_embedding])
    gen_sample = model(model_input)
    model_f = Model([noise, label], gen_sample, name="Generator")
    # define model
    return model_f


def combine_gan(g_model, d_model):
    # make weights in the discriminator not trainable
    d_model.trainable = False
    # get noise and label inputs from generator model
    gen_noise, gen_label = g_model.input
    # get image output from the generator model
    gen_output = g_model.output
    # connect image output and label input from generator as inputs to discriminator
    gan_output = d_model([gen_output, gen_label])

    # define gan model as taking noise and label and outputting a classification
    model = Model([gen_noise, gen_label], gan_output)
    # compile model
    opt = keras.optimizers.Adam(lr=1e-5)
    model.compile(loss='binary_crossentropy',  # we change loss function here. feature_matiching
                  optimizer=opt, metrics='accuracy')
    return model


def generate_latent_points2(noise_vec, n_samples, n_classes=2):
    # get random variables from a normal distribution with mu=0 and sigma=1.
    x_input = normal(0, 1, noise_vec * n_samples)
    # reshape into a batch of inputs for the network
    z_input = x_input.reshape(n_samples, noise_vec)
    # generate labels
    labels = randint(0, n_classes, n_samples)
    return [z_input, labels]


def train(g_model, d_model, gan_model, dataset, labels, laten_dim, n_epochs=10, n_batch=64):
    bat_per_epo = int(dataset.shape[0] / n_batch)
    half_batch = int(n_batch / 2)

    for i in range(n_epochs):
        # enumerate batches over the training set
        for j in range(bat_per_epo):
            # get randomly selected 'real' samples
            [X_real, labels_real], y_real = generate_real_samples2(
                dataset, labels, half_batch)
            # update discriminator model weights
            [X_fake, labels], y_fake = generate_fake_samples2(
                g_model, laten_dim, half_batch)
            d_loss1, _ = d_model.train_on_batch([X_real, labels_real], y_real)
            d_loss2, _ = d_model.train_on_batch([X_fake, labels], y_fake)
            # prepare points in latent space as input for the generator ##### z_input
            [z_input, labels_input] = generate_latent_points2(
                laten_dim, n_batch)
            y_gan = np.ones((n_batch, 1))
            _, g_loss = gan_model.train_on_batch(
                [z_input, labels_input], y_gan)

            # summarize loss on this batch
            print('>%d, %d/%d, d1=%.3f, d2=%.3f g=%.3f' %
                  (i+1, j+1, bat_per_epo, d_loss1, d_loss2, g_loss))
    # save the generator model
    g_model.save('cgan_generator2.h5')


def main():
    laten_dim = 50
    # create the discriminator
    d_model = define_discriminator()
    # create the generator
    g_model = define_generator(laten_dim)
    # create the gan
    gan_model = combine_gan(g_model, d_model)

    columns_ = df.columns[2:30]
    dataset = df[columns_]  # X
    labels = df.Class  # y.Class

    # train model
    print('dataset:', dataset)
    print(np.shape(dataset))

    train(g_model, d_model, gan_model, dataset, labels, laten_dim)


if __name__ == "__main__":
    main()
