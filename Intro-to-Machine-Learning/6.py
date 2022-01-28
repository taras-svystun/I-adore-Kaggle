import pandas as pd

from sklearn.metrics import mean_absolute_error as MAE
from sklearn.tree import DecisionTreeRegressor as DTR
from sklearn.model_selection import train_test_split as TTS
from sklearn.ensemble import RandomForestRegressor as RFR

def get_best_depth(train_X, train_y, val_X, val_y):
    best = 1
    min_mae = float("inf")
    for candidate in [5, 10, 50, 100, 500]:
        model = DTR(max_leaf_nodes=candidate, random_state=10)
        model.fit(train_X, train_y)
        prediction = model.predict(val_X)
        mae = MAE(val_y, prediction)
        if mae < min_mae:
            best, min_mae = candidate, mae
    return best

iowa_file_path = 'Data\\Intro-to-Machine-Learning\\train.csv'
df = pd.read_csv(iowa_file_path)

y = df.SalePrice
features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']
X = df[features]

train_X, val_X, train_y, val_y = TTS(X, y, random_state=10)

model1 = DTR(random_state=10)
model1.fit(train_X, train_y)
preds1 = model1.predict(val_X)
mae1 = MAE(val_y, preds1)

best_depth = get_best_depth(train_X, train_y, val_X, val_y)
model2 = DTR(max_leaf_nodes=best_depth, random_state=10)
model2.fit(train_X, train_y)
preds2 = model2.predict(val_X)
mae2 = MAE(val_y, preds2)

model3 = RFR(random_state=10)
model3.fit(train_X, train_y)
preds3 = model3.predict(val_X)
mae3 = MAE(val_y, preds3)

print(f"Regular DTR, mae = {mae1}")
print(f"Best depth DTR, mae = {mae2}")
print(f"RFR, mae = {mae3}")