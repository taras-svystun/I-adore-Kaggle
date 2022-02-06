import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

cancer_b_data = pd.read_csv('Data/Data-visualization/cancer_b.csv', index_col="Id")
cancer_m_data = pd.read_csv('Data/Data-visualization/cancer_m.csv', index_col="Id")

max_perim = cancer_b_data.iloc[0:5,3].max()
mean_radius = cancer_m_data.loc[842517, "Radius (mean)"]

sns.distplot(a=cancer_b_data.loc[:, 'Area (mean)'], kde=False)
sns.distplot(a=cancer_m_data.loc[:, 'Area (mean)'], kde=False)

sns.kdeplot(data=cancer_b_data.loc[:, 'Radius (worst)'], shade=True, label="B")
sns.kdeplot(data=cancer_m_data.loc[:, 'Radius (worst)'], shade=True, label="M")
plt.legend()