import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.losses import SparseCategoricalCrossentropy
from keras.optimizers import Adam
from sklearn.datasets import make_blobs

np.set_printoptions(precision=2)

print("Tensorflow version:", tf.__version__)

def my_softmax(z):
    ez = np.exp(z)
    a = ez/np.sum(ez)
    return a

classes = 4
m = 1000
center = [[-5, 2], [-2, -2], [1, 2], [5, -2]]
std = 1.0
X_train, y_train = make_blobs(n_samples=m, centers=center, cluster_std=std, random_state=30)

model = Sequential(
    [
        Dense(50, activation = "relu"),
        Dense(30, activation = "relu"),
        Dense(10, activation = 'relu'),
        Dense(4, activation = "linear")
    ]
)

model.compile(
    loss = SparseCategoricalCrossentropy(from_logits=True),
    optimizer=Adam(0.001),
)

model.fit(
    X_train, y_train,
    epochs=10
)

output = model.predict(X_train)

# for i in range(200):
#     print(f"{output[i]}, category: {np.argmax(output[i])}, y_train: {y_train[i]}")

for i in range(200):
    color = ["#ff0000", "#00ff00", "#0000ff", "#ffff00"]
    # if y_train[i] != np.argmax(output[i]):
    #     plt.scatter(X_train[i, 0], X_train[i, 1], marker="o", c="#000000")
    # else:
    plt.scatter(X_train[i, 0], X_train[i, 1], marker="o", c=color[y_train[i]])

plt.show()