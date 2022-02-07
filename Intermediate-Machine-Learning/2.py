import pandas as pd
from sklearn.model_selection import train_test_split as TTS
from sklearn.ensemble import RandomForestRegressor as RFR
from sklearn.metrics import mean_absolute_error as MAE
from sklearn.impute import SimpleImputer as SI

X_full = pd.read_csv(
    "Data/Intermediate-Machine-Learning/train.csv", index_col="Id")
X_test_full = pd.read_csv(
    "Data/Intermediate-Machine-Learning/test.csv", index_col="Id")

X_full.dropna(axis=0, subset=['SalePrice'], inplace=True)
y = X_full.SalePrice
X_full.drop(['SalePrice'], axis=1, inplace=True)

X = X_full.select_dtypes(exclude=['object'])
X_test = X_test_full.select_dtypes(exclude=['object'])
X_train, X_valid, y_train, y_valid = TTS(X, y,
                                         train_size=0.8, test_size=0.2,
                                         random_state=0)

X_train.shape
missing_values = X_train.isnull().sum()
missing_values[missing_values > 0].shape[0]


def score_dataset(X_train, X_valid, y_train, y_valid):
    model = RFR(n_estimators=100, random_state=0)
    model.fit(X_train, y_train)
    pred = model.predict(X_valid)
    return MAE(y_valid, pred)


missing_columns = [
    col for col in X_train.columns if X_train[col].isnull().any()]
reduced_X_train = X_train.drop(missing_columns, axis=1)
reduced_X_valid = X_valid.drop(missing_columns, axis=1)

print("MAE (Drop columns with missing values):")
print(score_dataset(reduced_X_train, reduced_X_valid, y_train, y_valid))

imputer = SI()
imputed_X_train = pd.DataFrame(imputer.fit_transform(X_train))
imputed_X_valid = pd.DataFrame(imputer.transform(X_valid))
imputed_X_train.columns = X_train.columns
imputed_X_valid.columns = X_valid.columns
print("MAE (Imputation):")
print(score_dataset(imputed_X_train, imputed_X_valid, y_train, y_valid))

# My model
imputer = SI(strategy="most_frequent")
final_X_train = pd.DataFrame(imputer.fit_transform(X_train))
final_X_valid = pd.DataFrame(imputer.transform(X_valid))
final_X_train.columns = X_train.columns
final_X_valid.columns = X_valid.columns

model = RFR(n_estimators=500, random_state=0, max_depth=100)
model.fit(final_X_train, y_train)
pred = model.predict(final_X_valid)
print("My approach mae:\n" + str(MAE(y_valid, pred)))

final_X_test = pd.DataFrame(imputer.transform(X_test))
preds_test = model.predict(final_X_test)
