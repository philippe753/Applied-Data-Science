import tensorflow as tf
from tensorflow import keras
from numpy.random import normal
from keras import layers
from keras.models import Model
import numpy as np
from numpy.random import randint

sample = ((tf.Tensor, np.array), np.array)


# df_10_per = df.sample(n=math.floor(len(df)/10))
# X, y = df_10_per.iloc[:, 1:-2], df_10_per.iloc[:, -1:]


def generate_real_samples(dataset, labels, n_samples):
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


def generate_fake_samples(g_model, latent_dim, n_samples):
    z_input, labels_input = generate_latent_points(latent_dim, n_samples)

    datapoints = g_model.predict([z_input, labels_input])
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


def discriminator(n_classes: int=2):  # in_shape=(1,28)
    gen_sample = layers.Input(shape=(28), name="gen_sample")
    label = layers.Input(shape=(1,), dtype='int32', name="label")
    label_embedding = layers.Flatten()(layers.Embedding(n_classes, 28)(label))
    model_input = layers.multiply([gen_sample, label_embedding])

    layer = layers.Dense(128, activation='relu')(model_input)
    layer = layers.Dropout(0.2)(layer)
    layer = layers.Dense(64, activation='relu')(layer)
    layer = layers.Dropout(0.2)(layer)
    layer = layers.Dense(32, activation='relu', name='intermediate_layer')(layer)
    final = layers.Dense(1, activation='sigmoid')(layer)

    model_dis = Model(inputs=[gen_sample, label],
                      outputs=final, name="Discriminator")

    opt = keras.optimizers.Adam(lr=1e-5)
    model_dis.compile(loss='binary_crossentropy',
                      optimizer=opt, metrics='accuracy')

    return model_dis


def generator(latent_dim, n_classes=2):

    noise = layers.Input(shape=(latent_dim,))
    label = layers.Input(shape=(1,), dtype='int32')

    label_embedding = layers.Flatten()(layers.Embedding(n_classes, latent_dim)(label))
    model_input = layers.multiply([noise, label_embedding])

    layer = layers.Dense(128, activation='relu')(model_input)
    layer = layers.Dropout(0.2)(layer)
    layer = layers.Dense(64, activation='relu', name='intermediate_layer')(layer)
    layer = layers.Dropout(0.2)(layer)
    final = layers.Dense(28, activation='tanh')(layer)

    model_f = Model([noise, label], final, name="Generator")
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
    return model


def generate_latent_points(noise_vec, n_samples, n_classes=2):
    # get random variables from a normal distribution with mu=0 and sigma=1.
    x_input = normal(0, 1, noise_vec * n_samples)
    # reshape into a batch of inputs for the network
    z_input = x_input.reshape(n_samples, noise_vec)
    # generate labels
    labels = randint(0, n_classes, n_samples)
    return [z_input, labels]


def main():
    print("This is a library, you should use it as that, rather than running it!")


if __name__ == "__main__":
    main()
