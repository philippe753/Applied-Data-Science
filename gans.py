from sklearn.linear_model import LinearRegression
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.models import load_model
import numpy as np
from numpy.random import rand, randn
from numpy.random import randint
import matplotlib.pyplot as plt
from keras.utils.vis_utils import plot_model

# Import balanced dataset
df_balanced = pd.read_csv('balanced_data.csv')
# Pick only the froud data
froud = df_balanced[df_balanced["Class"] == 1]
# pd datatrame of only the Vs
froud_V_colums = froud.columns[2:30]
df_froud_Vs = froud[froud_V_colums]
print(df_froud_Vs.head())  # froun v1 to v28


# Select real samples


def generate_real_samples(dataset, n_samples):
    # Random pick instances from the dataset
    idx = randint(0, dataset.shape[0], n_samples)
    # retrieve selected images
    X = dataset.iloc[idx].values
    X = tf.reshape(X, [n_samples, 28])
    # print("pick real")
    # print(np.shape(X))
    # print(type(X))
    # generate 'real' class labels (1)
    y = np.ones((n_samples, 1))
    return X, y

# generate n fake samples with class labels


# def generate_fake_samples(n_samples):
#     # generate uniform random numbers in [0,1]
#     X = rand(28 * n_samples)
#     # reshape into a batch of grayscale images
#     X = tf.reshape(X, [n_samples, 28])
#     # generate 'fake' class labels (0)
#     y = np.zeros((n_samples, 1))
#     return X, y


def discriminator_model(in_shape=(1, 28)):
    model = keras.Sequential(
        [
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(32, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(1, activation='sigmoid')
        ]
    )
    # Compile model
    opt = keras.optimizers.Adam(lr=1e-4)
    model.compile(loss='binary_crossentropy',
                  optimizer=opt, metrics='accuracy')
    return model


def generator_model(latent_dim):
    model = keras.Sequential([
        layers.Dense(128, activation='relu', input_dim=latent_dim),
        layers.Dropout(0.2),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(28, activation='sigmoid')
    ])
    return model

# define the combined generator and discriminator model, for updating the generator


def define_gan(g_model, d_model):
    # make weights in the discriminator not trainable
    d_model.trainable = False
    # connect them
    model = keras.Sequential()
    # add generator
    model.add(g_model)
    # add the discriminator
    model.add(d_model)
    # compile model
    opt = keras.optimizers.Adam(lr=1e-4)
    model.compile(loss='binary_crossentropy',
                  optimizer=opt, metrics='accuracy')
    return model


# generate points in latent space as input for the generator
def generate_latent_points(latent_dim, n_samples):
    # generate points in the latent space
    x_input = randn(latent_dim * n_samples)
    # reshape into a batch of inputs for the network
    x_input = x_input.reshape(n_samples, latent_dim)
    return x_input


# use the generator to generate n fake examples, with class labels
def generate_fake_samples(g_model, latent_dim, n_samples):
    # generate points in latent space
    x_input = generate_latent_points(latent_dim, n_samples)
    # predict outputs
    X = g_model.predict(x_input)
    # create 'fake' class labels (0)
    y = np.zeros((n_samples, 1))
    return X, y


# evaluate the discriminator, plot generated images, save generator model
def summarize_performance(epoch, g_model, d_model, dataset, latent_dim, n_samples=100):
    # prepare real samples
    X_real, y_real = generate_real_samples(dataset, n_samples)
    # evaluate discriminator on real examples
    _, acc_real = d_model.evaluate(X_real, y_real, verbose=0)
    # prepare fake examples
    x_fake, y_fake = generate_fake_samples(g_model, latent_dim, n_samples)
    # evaluate discriminator on fake examples
    _, acc_fake = d_model.evaluate(x_fake, y_fake, verbose=0)
    # summarize discriminator performance
    print('>Accuracy real: %.0f%%, fake: %.0f%%' %
          (acc_real*100, acc_fake*100))
    # save the generator model tile file
    filename = 'generator_model_%03d.h5' % (epoch + 1)
    g_model.save(filename)

 # train the generator and discriminator


def train(g_model, d_model, gan_model, dataset, latent_dim, n_epochs=100, n_batch=64):
    bat_per_epo = int(dataset.shape[0] / n_batch)

    half_batch = int(n_batch / 2)
    # manually enumerate epochs
    for i in range(n_epochs):
        # enumerate batches over the training set
        for j in range(bat_per_epo):
            # get randomly selected 'real' samples
            X_real, y_real = generate_real_samples(dataset, half_batch)
            # generate 'fake' examples
            X_fake, y_fake = generate_fake_samples(
                g_model, latent_dim, half_batch)
            # create training set for the discriminator
            X, y = np.vstack((X_real, X_fake)), np.vstack((y_real, y_fake))
            # update discriminator model weights
            d_loss, _ = d_model.train_on_batch(X, y)
            # prepare points in latent space as input for the generator
            X_gan = generate_latent_points(latent_dim, n_batch)
            # create inverted labels for the fake samples
            y_gan = np.ones((n_batch, 1))
            # update the generator via the discriminator's error
            _, g_loss = gan_model.train_on_batch(X_gan, y_gan)
            # summarize loss on this batch
            print('>%d, %d/%d, d=%.3f, g=%.3f' %
                  (i+1, j+1, bat_per_epo, d_loss, g_loss))
        # evaluate the model performance, sometimes
        if (i+1) % 10 == 0:
            summarize_performance(i, g_model, d_model, dataset, latent_dim)


# # train the discriminator model
# def train_discriminator(model, dataset, n_iter=100, n_batch=256):
#     half_batch = int(n_batch / 2)
#     # manually enumerate epochs
#     for i in range(n_iter):
#         # generate 'fake' examples
#         X_fake, y_fake = generate_fake_samples(half_batch)
#         # update discriminator on fake samples
#         _, fake_acc = model.train_on_batch(X_fake, y_fake)
#         # get randomly selected 'real' samples
#         X_real, y_real = generate_real_samples(dataset, half_batch)
#         # update discriminator on real samples
#         _, real_acc = model.train_on_batch(X_real, y_real)

#         # summarize performance
#         print('>%d real=%.0f%% fake=%.0f%%' %
#               (i+1, real_acc*100, fake_acc*100))


# # define the size of the latent space
# latent_dim = 50
# # define the generator model
# model = generator_model(latent_dim)
# summarize the GENERATOR model
# model.summary()
# plot the model
# plot_model(model, to_file='generator_plot.png',
#            show_shapes=True, show_layer_names=True)


# # define the discriminator model
# model = discriminator_model()

# dataset = df_froud_Vs

# fit the model
# train_discriminator(model, dataset)


# # size of the latent space
# latent_dim = 50
# # define the discriminator model
# model = generator_model(latent_dim)
# # generate samples
# n_samples = 25
# X, _ = generate_fake_samples(model, latent_dim, n_samples)
# print(X, _)


# # size of the latent space
# latent_dim = 100
# # create the discriminator
# d_model = discriminator_model()
# # create the generator
# g_model = generator_model(latent_dim)
# # create the gan
# gan_model = define_gan(g_model, d_model)


# #size of the latent space
# latent_dim = 50
# # create the discriminator
# d_model = discriminator_model()
# # create the generator
# g_model = generator_model(latent_dim)
# # create the gan
# gan_model = define_gan(g_model, d_model)
# # load image data
# dataset = df_froud_Vs
# # train model
# train(g_model, d_model, gan_model, dataset, latent_dim)


# Import model
model = load_model('generator_model_100.h5')
num_new_samples = 250
# generate images
# generates 25 samples.
latent_points = generate_latent_points(50, num_new_samples)
# generate images
X = model.predict(latent_points)
print(X)

plt.plot(X[:, 0])
plt.show()
print("Done")


# # Linear regression without extra frouds samples.

X = df_balanced[df_balanced.columns[2:30]]  # v1 to v28
y = df_balanced.Class

reg = LinearRegression().fit(X, y)
print("Linear regression score: ")
print(reg.score(X, y))


# Linear regression without extra frouds samples.
print(X)
X = pd.concat([X, pd.DataFrame(latent_points)])
y = pd.concat([y, pd.DataFrame(np.ones((1, num_new_samples)))])
print(X)

reg = LinearRegression().fit(X, y)
print("Linear regression score (extra froud samples): ")
print(reg.score(X, y))
