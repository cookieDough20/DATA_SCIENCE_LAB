import pandas as pd
from sklearn.datasets import load_iris

iris=load_iris()
df=pd.DataFrame(iris.data)

print(df.mean())
print(df.median())
print(df.mode().iloc[0])
print(df.var())
print(df.std())
