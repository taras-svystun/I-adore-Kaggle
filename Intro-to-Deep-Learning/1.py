from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd

red_wine = pd.read_csv('Data/Intro-to-Deep-Learning/red-wine.csv')

plt.style.use('seaborn-whitegrid')
# Set Matplotlib defaults
plt.rc('figure', autolayout=True)
plt.rc('axes', labelweight='bold', labelsize='large',
       titleweight='bold', titlesize=18, titlepad=10)

model = keras.Sequential([
    layers.Dense(units=1, input_shape=[3])
])

print(red_wine.shape)
# The target is 'quality'
input_shape = [red_wine.shape[1] - 1]
model = keras.Sequential([
    layers.Dense(units=1, input_shape=input_shape)
])
w, b = model.weights
print(w, b)

# Additional
model = keras.Sequential([
    layers.Dense(units=1, input_shape=[1])
])
x = tf.linspace(-3, 3, 100)
y = model.predict(x)

plt.figure()
plt.plot(x, y)
plt.xlim(-3, 3)
plt.ylim(-3, 3)
plt.xlabel("Input x")
plt.ylabel("Predicted y")
w, b = model.weights
plt.title(f"Weight : {round(float(w[0][0]), 3)}\nBias : {b[0]}")
plt.show()
