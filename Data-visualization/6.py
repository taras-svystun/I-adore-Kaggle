import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

spotify_data = pd.read_csv('Data/Data-visualization/spotify.csv',
index_col="Date", parse_dates=True)

sns.set_style("darkgrid")
plt.figure(figsize=(9 ,6))
sns.lineplot(data=spotify_data)
plt.show()