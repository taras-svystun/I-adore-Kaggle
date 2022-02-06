from cProfile import label
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

path = "Data/Data-visualization/museum_visitors.csv"
museum_data = pd.read_csv(path, index_col="Date", parse_dates=True)

ca_museum_jul18 = museum_data.loc["2018-07-01","Chinese American Museum"]
avila_oct18 = museum_data.loc["2018-10-01","Avila Adobe"] - museum_data.loc["2018-10-01","Firehouse Museum"]

# museum_data.plot()
plt.title("Museum data")
# sns.lineplot(data=museum_data)

museum_data["Avila Adobe"].plot(label="Avila Adobe visitors")
plt.xlabel("Date")
plt.show()





# plt.figure(figsize=(14,6))

# plt.title("Daily Global Streams of Popular Songs in 2017-2018")
# sns.lineplot(data=spotify_data['Shape of You'], label="Shape of You")
# sns.lineplot(data=spotify_data['Despacito'], label="Despacito")
# plt.xlabel("Date")