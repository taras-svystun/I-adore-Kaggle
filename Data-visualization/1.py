import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# The next line is realy optional
# plt.figure(figsize=(10,6))
fifa = pd.read_csv("Data/Data-visualization/fifa.csv")
print(fifa.head())
sns.lineplot(data=fifa)
plt.show()