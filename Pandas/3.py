import pandas as pd

pd.set_option("display.max_rows", 5)
reviews = pd.read_csv("Data/Pandas/winemag-data-130k-v2.csv", index_col=0)

median_points = reviews.points.median()

countries = reviews.country.unique()

reviews_per_country = reviews.country.value_counts()

centered_price = reviews.price - reviews.price.mean()

max_idx = (reviews.points / reviews.price).idxmax()
bargain_wine = reviews.loc[max_idx, 'title']

# tropical = reviews['description'].str.contains('tropical').value_counts()[True]
# fruity = reviews['description'].str.contains('fruity').value_counts()[True]
# descriptor_counts = pd.Series({"tropical": tropical, "fruity": fruity})
tropical = reviews.description.map(lambda description: "tropical" in description).sum()
fruity = reviews.description.map(lambda description: "fruity" in description).sum()
descriptor_counts = pd.Series({"tropical": tropical, "fruity": fruity})

def star(row):
    if row.points >= 95 or row.country == "Canada":
        return 3
    elif 95 > row.points >= 85:
        return 2
    return 1

star_ratings = reviews.apply(star, axis="columns")
print(star_ratings)