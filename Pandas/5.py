import pandas as pd

pd.set_option("display.max_rows", 5)
reviews = pd.read_csv("Data/Pandas/winemag-data-130k-v2.csv", index_col=0)

dtype = reviews.points.dtype
point_strings = reviews.points.map(lambda x: str(x))
n_missing_prices = reviews.price.isna().sum()
replaced = reviews.region_1.fillna("Unknown")
reviews_per_region = replaced.value_counts()