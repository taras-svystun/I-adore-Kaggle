import pandas as pd

df_dict = {
    "Yes": [11, 12],
    "No": [-7, -17]
}

df = pd.DataFrame(df_dict)
print(df)

df_dict2 = {'Bob': ['I liked it.', 'It was awful.'], 'Sue': ['Pretty good.', 'Bland.']}
df2 = pd.DataFrame(df_dict2, index=["first", "second"])
print(df2)

df3 = pd.Series([30, 35, 40], index=['2015 Sales', '2016 Sales', '2017 Sales'], name='Product A')
print(df3)

df_dict = {"Apples": 30, "Bananas": 21}
fruits = pd.DataFrame(df_dict, index=[0])
print(fruits)

df2_dict = {"Apples": [35, 41], "Bananas": [21, 34]}
fruit_sales = pd.DataFrame(df2_dict, index=["2017 Sales", "2018 Sales"])

ingredients = pd.Series("4 cups$1 cup$2 large$1 can".split("$"), index="Flour$Milk$Eggs$Spam".split("$"), name = "Dinner")

# reviews = pd.read_csv("../input/wine-reviews/winemag-data_first150k.csv", index_col=0)

animals = pd.DataFrame({'Cows': [12, 20], 'Goats': [22, 19]}, index=['Year 1', 'Year 2'])
animals.to_csv("Data\\Pandas\\cows_and_goats.csv")
