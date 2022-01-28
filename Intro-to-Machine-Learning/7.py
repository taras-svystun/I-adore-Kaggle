import pandas as pd
from sklearn.ensemble import RandomForestRegressor as RFR
from sklearn.metrics import mean_absolute_error as MAE
from sklearn.model_selection import train_test_split as TTS

iowa_file_path = 'Data\\Intro-to-Machine-Learning\\train.csv'
home_data = pd.read_csv(iowa_file_path)
y = home_data.SalePrice

features = ['LotArea', 'YearBuilt', '1stFlrSF',
            '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']

X = home_data[features]
X.head()

train_X, val_X, train_y, val_y = TTS(X, y, random_state=1)

rf_model = RFR(random_state=1)
rf_model.fit(train_X, train_y)
rf_val_predictions = rf_model.predict(val_X)
rf_val_mae = MAE(rf_val_predictions, val_y)

print("Validation MAE for Random Forest Model: {:,.0f}".format(rf_val_mae))

rf_model_on_full_data = RFR(random_state=10)
rf_model_on_full_data.fit(X, y)
# test_data_path = '../input/test.csv'

# test_data = pd.read_csv(test_data_path)
# test_X = test_data[features]

# test_preds = rf_model_on_full_data.predict(test_X)
