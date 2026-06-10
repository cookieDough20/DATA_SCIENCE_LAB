import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

iris=load_iris()
df=pd.DataFrame(iris.data,columns=iris.feature_names)

sns.histplot(df["sepal length (cm)"])
plt.show()

sns.boxplot(x=df["sepal length (cm)"])
plt.show()

sns.scatterplot(x=df["sepal length (cm)"],y=df["sepal width (cm)"])
plt.show()
