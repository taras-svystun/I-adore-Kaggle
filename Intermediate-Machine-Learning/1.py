import pandas as pd
from sklearn.model_selection import train_test_split as TTS
from sklearn.ensemble import RandomForestRegressor as RFR
from sklearn.metrics import mean_absolute_error as MAE

X_full = pd.read_csv(
    "Data/Intermediate-Machine-Learning/train.csv", index_col="Id")
X_test_full = pd.read_csv(
    "Data/Intermediate-Machine-Learning/test.csv", index_col="Id")

y = X_full.SalePrice

features = ['LotArea', 'YearBuilt', '1stFlrSF',
            '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']
X = X_full[features].copy()
X_test = X_test_full[features].copy()

X_train, X_valid, y_train, y_valid = TTS(X, y,
                                         train_size=0.8, test_size=0.2, random_state=0)

model_1 = RFR(n_estimators=50, random_state=0)
model_2 = RFR(n_estimators=100, random_state=0)
model_3 = RFR(n_estimators=100, criterion='absolute_error', random_state=0)
model_4 = RFR(n_estimators=200, min_samples_split=20, random_state=0)
model_5 = RFR(n_estimators=100, max_depth=7, random_state=0)
models = [model_1, model_2, model_3, model_4, model_5]


def score_model(model, x_t=X_train, x_v=X_valid, y_t=y_train, y_v=y_valid):
    model.fit(x_t, y_t)
    pred = model.predict(x_v)
    mae = MAE(y_v, pred)
    return mae

# for i, model in enumerate(models):
    # print(f"Model {i + 1} mae = {round(score_model(model))}")


my_model = RFR(n_estimators=100, criterion='absolute_error', random_state=0)
my_model.fit(X, y)
pred = my_model.predict(X_test)
output = pd.DataFrame({'Id': X_test.index,
                       'SalePrice': pred})
