import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np
import seaborn as sns

data = pd.read_csv('iris.csv')
data.head()
x = data.iloc[:, [0, 1, 2, 3]].values


iris_outcome = pd.crosstab(index=data["species"], columns="count")
iris_setosa=data.loc[data["species"]=="Iris-setosa"]
iris_virginica=data.loc[data["species"]=="Iris-virginica"]
iris_versicolor=data.loc[data["species"]=="Iris-versicolor"]


sns.FacetGrid(data,hue="species",height=3).map(sns.distplot,"petal_length").add_legend()
sns.FacetGrid(data,hue="species",height=3).map(sns.distplot,"petal_width").add_legend()
sns.FacetGrid(data,hue="species",height=3).map(sns.distplot,"sepal_length").add_legend()
plt.show()


X = data.iloc[:,:-1]
Y = data.iloc[:,-1]
X.head()

Y.unique()

ss = StandardScaler()
X = ss.fit_transform(X)
X = pd.DataFrame(X, columns=data.columns[:-1])
X.head()

pca = PCA(n_components=2)
X = pca.fit_transform(X)
X = pd.DataFrame(X, columns=['PC1', 'PC2'])
X.head()

plt.figure(figsize=(8,6))
plt.scatter(X['PC1'], X['PC2'])
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('Scatter plot of original data')
plt.show()

ssd = []
for k in range(1,11):
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(X)
    ssd.append(kmeans.inertia_)

plt.plot(range(1,11), ssd)
plt.xticks(range(1,11))
plt.xlabel('Number of clusters')
plt.ylabel('Sum of squares')
plt.title('Elbow method to find optimal K')
plt.show()

k_means = KMeans(n_clusters=3)
k_means_model = k_means.fit(X)

fig = plt.figure(figsize=(14,6))
#predicted
ax1 = fig.add_subplot(121)
pred = k_means_model.predict(X)
colors = ['green','blue','orange']
for idx in range(3):
    plt.scatter(X[pred == idx]['PC1'],X[pred == idx]['PC2'], c = colors[idx])
plt.scatter(k_means_model.cluster_centers_[:, 0], k_means_model.cluster_centers_[:, 1], s=100, c='black', marker='o', label='Centroids')

plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('Predicted Clusters with Centroids')

#actual
ax2 = fig.add_subplot(122)
plt.scatter(X[Y == 'setosa']['PC1'],X[Y == 'setosa']['PC2'], c='blue', label='setosa')
plt.scatter(X[Y == 'versicolor']['PC1'],X[Y == 'versicolor']['PC2'], c='green', label='versicolor')
plt.scatter(X[Y == 'virginica']['PC1'],X[Y == 'virginica']['PC2'], c='orange', label='virginica')
plt.scatter(k_means_model.cluster_centers_[:, 0], k_means_model.cluster_centers_[:, 1], s=100, c='black', marker='o', label='Centroids')
plt.legend()
plt.legend()
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('Actual Scatter plot of data')
plt.show()
