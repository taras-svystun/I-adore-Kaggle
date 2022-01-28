import pandas as pd
from sklearn.metrics import mean_absolute_error as MAE
from sklearn.model_selection import train_test_split as TTS
from sklearn.tree import DecisionTreeRegressor as DTR


def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):
    model = DTR(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(train_X, train_y)
    preds_val = model.predict(val_X)
    mae = MAE(val_y, preds_val)
    return(mae)


iowa_file_path = 'Data\\Intro-to-Machine-Learning\\train.csv'
home_data = pd.read_csv(iowa_file_path)
y = home_data.SalePrice
features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']
X = home_data[features]

train_X, val_X, train_y, val_y = TTS(X, y, random_state=1)

iowa_model = DTR(random_state=1)
iowa_model.fit(train_X, train_y)

val_predictions = iowa_model.predict(val_X)
val_mae = MAE(val_predictions, val_y)
print("Validation MAE: {:,.0f}".format(val_mae))


candidate_max_leaf_nodes = [5, 25, 50, 100, 250, 500]
best_tree_size = candidate_max_leaf_nodes[0]
min_mae = float("inf")
for candidate in candidate_max_leaf_nodes:
    mae = get_mae(candidate, train_X, val_X, train_y, val_y)
    if mae < min_mae:
        best_tree_size, min_mae = candidate, mae

print(best_tree_size)

final_model = DTR(max_leaf_nodes = best_tree_size)
final_model.fit(X, y)

