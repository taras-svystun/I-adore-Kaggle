import pandas as pd
from sklearn.tree import DecisionTreeRegressor as DTR
from sklearn.model_selection import train_test_split as TTS
from sklearn.metrics import mean_absolute_error as MAE


iowa_file_path = 'Data\\Intro-to-Machine-Learning\\train.csv'
home_data = pd.read_csv(iowa_file_path)
y = home_data.SalePrice
feature_columns = ['LotArea', 'YearBuilt', '1stFlrSF',
                   '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']
X = home_data[feature_columns]

train_X, val_X, train_y, val_y = TTS(X, y, random_state=1)
iowa_model = DTR(random_state=1)
iowa_model.fit(train_X, train_y)

val_predictions = iowa_model.predict(val_X)

print(val_predictions[:5])
print(val_y.head())

val_mae = MAE(val_y, val_predictions)
print(val_mae)
