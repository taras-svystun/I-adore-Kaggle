import pandas as pd

pd.set_option("display.max_rows", 5)
reviews = pd.read_csv("Data/Pandas/winemag-data-130k-v2.csv", index_col=0)
print(reviews.head())

renamed = reviews.rename(columns={
    "region_1": "region",
    "region_2": "locale"
})
reindexed = reviews.rename_axis("wines", axis="rows")

gaming_products = pd.read_csv("../input/things-on-reddit/top-things/top-things/reddits/g/gaming.csv")
gaming_products['subreddit'] = "r/gaming"
movie_products = pd.read_csv("../input/things-on-reddit/top-things/top-things/reddits/m/movies.csv")
movie_products['subreddit'] = "r/movies"
combined_products = pd.concat([gaming_products, movie_products])

powerlifting_meets = pd.read_csv("../input/powerlifting-database/meets.csv")
powerlifting_competitors = pd.read_csv("../input/powerlifting-database/openpowerlifting.csv")
powerlifting_meets = powerlifting_meets.set_index("MeetID")
powerlifting_competitors = powerlifting_competitors.set_index("MeetID")
powerlifting_combined = powerlifting_meets.join(powerlifting_competitors)
