import pandas as pd

iowa_file_path = 'Data\\Intro-to-Machine-Learning\\train.csv'
home_data = pd.read_csv(iowa_file_path)
print(home_data.describe())

avg_lot_size = round(home_data["LotArea"].mean())
newest_home_age = 2022 - home_data["YearBuilt"].max()
print(newest_home_age)