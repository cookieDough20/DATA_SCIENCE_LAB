import seaborn as sns

df=sns.load_dataset("titanic")

df["age"].fillna(df["age"].median(),inplace=True)
df.drop_duplicates(inplace=True)

print(df.head())
print(df.describe())
