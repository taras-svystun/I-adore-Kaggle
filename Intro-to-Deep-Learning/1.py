from tensorflow import keras
from tensorflow.keras import layers
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
