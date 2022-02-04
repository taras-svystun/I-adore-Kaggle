import pandas as pd

pd.set_option("display.max_rows", 5)
reviews = pd.read_csv("Data/Pandas/winemag-data-130k-v2.csv", index_col=0)

# reviews.groupby('points').points.count()
# reviews.groupby('country').price.mean().max()

# print(reviews.groupby('country').apply(lambda df: df.title.iloc[0]))
# print(reviews.groupby(['country', 'points']).apply(lambda df: df.loc[df.points.idxmin()]).loc["Ukraine"])

# print(reviews.groupby(["points"]).price.agg([min, max]))

# countries_reviewed = reviews.groupby(["country", "province"]).description.agg([len])
# countries_reviewed.reset_index()
# print(countries_reviewed.sort_values(by='len', ascending=False))
# print(countries_reviewed.sort_index())
# countries_reviewed.sort_values(by=['country', 'len'])

print(reviews.head())

reviews_written = reviews.groupby("taster_twitter_handle").taster_twitter_handle.count()
best_rating_per_price = reviews.groupby("price")["points"].max()
price_extremes = reviews.groupby("variety").price.agg([min, max])
sorted_varieties = price_extremes.sort_values(by=["min", "max"], ascending=False)
reviewer_mean_ratings = reviews.groupby("taster_name").apply(lambda df: df.points.mean())
reviewer_mean_ratings.describe()
country_variety_counts = reviews.groupby(["country", "variety"]).size().sort_values(ascending=False)
