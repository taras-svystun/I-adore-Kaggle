import pandas as pd


reviews = pd.read_csv("Data/Pandas/winemag-data-130k-v2.csv", index_col=0)
print(reviews.head())
desc = reviews["description"]
first_description = reviews.description.iloc[0]
first_row = reviews.iloc[0, :]
first_descriptions = reviews.loc[0:9, "description"]
sample_reviews = reviews.loc[[1, 2, 3, 5, 8]]
df = reviews.loc[[0, 1, 10, 100], "country$province$region_1$region_2".split("$")]
df = reviews.loc[:99, "country$variety".split("$")]
italian_wines = reviews.loc[ reviews.country == "Italy"]
top_oceania_wines = reviews.loc[ (reviews.country.isin(["Australia", "New Zealand"])) & (reviews.points > 94)]
