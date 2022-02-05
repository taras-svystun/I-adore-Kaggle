from pickletools import optimize
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.compose import make_column_transformer as MCT
from sklearn.compose import make_column_selector as MCS
from sklearn.preprocessing import StandardScaler as SS
from sklearn.preprocessing import OneHotEncoder as OHE
from tensorflow import keras
from tensorflow.keras import layers

plt.style.use('seaborn-whitegrid')
# Set Matplotlib defaults
plt.rc('figure', autolayout=True)
plt.rc('axes', labelweight='bold', labelsize='large',
       titleweight='bold', titlesize=18, titlepad=10)
plt.rc('animation', html='html5')

fuel = pd.read_csv("Data/Intro-to-Deep-Learning/fuel.csv")

X = fuel.copy()
y = X.pop("FE")

preprocessor = MCT(
    (SS(),
    MCS(dtype_include=np.number)),
    (OHE(sparse=False),
    MCS(dtype_include=object))
)

X = preprocessor.fit_transform(X)
y = np.log(y)
input_shape = [X.shape[1]]

model = keras.Sequential([
    layers.Dense(128, activation="relu", input_shape=input_shape),
    layers.Dense(128, activation="selu", input_shape=input_shape),
    layers.Dense(64, activation="elu", input_shape=input_shape),
    layers.Dense(1)
])

model.compile(
    optimizer="adam",
    loss="mse"
)

history = model.fit(
    X, y,
    batch_size=128,
    epochs=128
)
history_df = pd.DataFrame(history.history)
print(history_df.head())
history_df.loc[:,"loss"].plot()
plt.show()
