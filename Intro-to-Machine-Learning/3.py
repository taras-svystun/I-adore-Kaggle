import pandas as pd
from sklearn.tree import DecisionTreeRegressor as DTR

iowa_file_path = 'Data\\Intro-to-Machine-Learning\\train.csv'
df = pd.read_csv(iowa_file_path)

print(df.columns)
y = df.SalePrice
features = ["LotArea", "YearBuilt", "1stFlrSF", "2ndFlrSF", "FullBath", "BedroomAbvGr", "TotRmsAbvGrd"]
x = df[features]

print(x.describe())
print(x.head())

iowa_model = DTR(random_state = 1)
iowa_model.fit(x, y)

predictions = iowa_model.predict(x)
print(predictions)

print("Expected:")
print(y.head())
print()
print("Actual:")
print(iowa_model.predict(x.head()))