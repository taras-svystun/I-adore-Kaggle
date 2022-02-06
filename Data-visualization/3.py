import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

ign_data = pd.read_csv("Data/Data-visualization/ign_scores.csv", index_col="Platform")

high_score = ign_data.loc["PC"].max()
worst_genre = ign_data.loc["PlayStation Vita"].idxmin()

sns.barplot(x=ign_data.index, y=ign_data["Racing"])
# plt.show()
sns.heatmap(data=ign_data, annot=True)
# plt.show()
