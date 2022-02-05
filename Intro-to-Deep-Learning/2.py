from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd

concrete = pd.read_csv("Data/Intro-to-Deep-Learning/concrete.csv")

plt.style.use('seaborn-whitegrid')
# Set Matplotlib defaults
plt.rc('figure', autolayout=True)
plt.rc('axes', labelweight='bold', labelsize='large',
       titleweight='bold', titlesize=18, titlepad=10)

print(concrete.head())
input_shape = [concrete.shape[1] - 1]

model = keras.Sequential([
    layers.Dense(units=512, activation="relu", input_shape=input_shape),
    layers.Dense(units=512, activation="relu"),
    layers.Dense(units=512, activation="relu"),
    layers.Dense(units=1)
])

activation_layer = layers.Activation('swish')

x = tf.linspace(-3.0, 3.0, 100)
y = activation_layer(x) 

plt.figure()
plt.plot(x, y)
plt.xlim(-3, 3)
plt.xlabel("Input")
plt.ylabel("Output")
plt.show()