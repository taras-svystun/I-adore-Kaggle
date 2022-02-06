import pandas as pd
import seaborn as sns

candy_data = pd.read_csv("Data/Data-visualization/candy.csv", index_col="id")
print(candy_data[candy_data["competitorname"].isin(['3 Musketeers', 'Almond Joy'])]['winpercent'])
sns.scatterplot(data=candy_data, x='sugarpercent', y='winpercent')
sns.regplot(data=candy_data, x='sugarpercent', y='winpercent')
sns.scatterplot(data=candy_data, x='sugarpercent', y='winpercent', hue='chocolate')
sns.lmplot(data=candy_data, x='pricepercent', y='winpercent', hue='chocolate')
sns.swarmplot(data=candy_data, x='chocolate', y='winpercent')
