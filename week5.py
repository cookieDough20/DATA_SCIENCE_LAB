import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

df=sns.load_dataset("tips")
X=df.select_dtypes(include=["float64","int64"])

X=StandardScaler().fit_transform(X)

pca=PCA(n_components=2)
result=pca.fit_transform(X)

print(result[:5])
