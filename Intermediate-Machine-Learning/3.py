import pandas as pd
from sklearn.model_selection import train_test_split as TTS
from sklearn.ensemble import RandomForestRegressor as RFR
from sklearn.metrics import mean_absolute_error as MAE
from sklearn.preprocessing import OrdinalEncoder as OE
from sklearn.preprocessing import OneHotEncoder as OHE

X = pd.read_csv('../input/train.csv', index_col='Id')
X_test = pd.read_csv('../input/test.csv', index_col='Id')
X.dropna(axis=0, subset=['SalePrice'], inplace=True)
y = X.SalePrice
X.drop(['SalePrice'], axis=1, inplace=True)
cols_with_missing = [col for col in X.columns if X[col].isnull().any()]
X.drop(cols_with_missing, axis=1, inplace=True)
X_test.drop(cols_with_missing, axis=1, inplace=True)
X_train, X_valid, y_train, y_valid = TTS(X, y,
                                         train_size=0.8, test_size=0.2,
                                         random_state=0)

def score_dataset(X_train, X_valid, y_train, y_valid):
    model = RFR(n_estimators=100, criterion="absolute_error", random_state=1)
    model.fit(X_train, y_train)
    pred = model.predict(X_valid)
    return MAE(y_valid, pred)

drop_X_train = X_train.select_dtypes(exclude="object")
drop_X_valid = X_valid.select_dtypes(exclude="object")


"""
data = pd.read_csv('../input/melbourne-housing-snapshot/melb_data.csv')
y = data.Price
X = data.drop(['Price'], axis=1)
X_train_full, X_valid_full, y_train, y_valid = TTS(X, y, train_size=0.8, test_size=0.2,
                                                   random_state=0)
cols_with_missing = [
    col for col in X_train_full.columns if X_train_full[col].isnull().any()]
X_train_full.drop(cols_with_missing, axis=1, inplace=True)
X_valid_full.drop(cols_with_missing, axis=1, inplace=True)
low_cardinality_cols = [cname for cname in X_train_full.columns if X_train_full[cname].nunique() < 10 and
                        X_train_full[cname].dtype == "object"]
numerical_cols = [
    cname for cname in X_train_full.columns if X_train_full[cname].dtype in ['int64', 'float64']]
my_cols = low_cardinality_cols + numerical_cols
X_train = X_train_full[my_cols].copy()
X_valid = X_valid_full[my_cols].copy()
# Get list of categorical variables
s = (X_train.dtypes == 'object')
object_cols = list(s[s].index)

def score_dataset(X_train, X_valid, y_train, y_valid):
    model = RFR(n_estimators=100, criterion="absolute_error", random_state=1)
    model.fit(X_train, y_train)
    pred = model.predict(X_valid)
    return MAE(y_valid, pred)

# 1 - drop
drop_X_train = X_train.select_dtypes(exclude=['object'])
drop_X_valid = X_valid.select_dtypes(exclude=['object'])
print("MAE from Approach 1 (Drop categorical variables):")
print(score_dataset(drop_X_train, drop_X_valid, y_train, y_valid))

# Ordinal encoder
label_X_train = X_train.copy()
label_X_valid = X_valid.copy()
ordinal_encoder = OE()
label_X_train[object_cols] = ordinal_encoder.fit_transform(X_train[object_cols])
label_X_valid[object_cols] = ordinal_encoder.transform(X_valid[object_cols])
print("MAE from Approach 2 (Ordinal Encoding):") 
print(score_dataset(label_X_train, label_X_valid, y_train, y_valid))

# 3 One Hot Encoder
OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
OH_cols_train = pd.DataFrame(OH_encoder.fit_transform(X_train[object_cols]))
OH_cols_valid = pd.DataFrame(OH_encoder.transform(X_valid[object_cols]))
OH_cols_train.index = X_train.index
OH_cols_valid.index = X_valid.index
num_X_train = X_train.drop(object_cols, axis=1)
num_X_valid = X_valid.drop(object_cols, axis=1)
OH_X_train = pd.concat([num_X_train, OH_cols_train], axis=1)
OH_X_valid = pd.concat([num_X_valid, OH_cols_valid], axis=1)
print("MAE from Approach 3 (One-Hot Encoding):") 
print(score_dataset(OH_X_train, OH_X_valid, y_train, y_valid))

"""
