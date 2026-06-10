import pandas as pd
import numpy as np

df = pd.DataFrame({
    "Age":[25,30,np.nan,40],
    "Dept":["HR","IT",None,"Finance"]
})

df["Age"].fillna(df["Age"].mean(), inplace=True)
df["Dept"].fillna(df["Dept"].mode()[0], inplace=True)

df.drop_duplicates(inplace=True)
print(df)
