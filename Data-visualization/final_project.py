import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

path = 'Data/Data-visualization/Earthquakes.csv'

df = pd.read_csv(path)
print(df.shape)

sns.lineplot(data=df.loc[:200,'MAGNITUDE'])
plt.show()