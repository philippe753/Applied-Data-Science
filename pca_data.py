from __future__ import annotations
from cProfile import label
from turtle import color
from matplotlib import markers, style

import numpy as npgit
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import statistics as stats
from scipy.stats import norm
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import seaborn as sns

credit_card_data=pd.read_csv(r'C:\Users\tmara\Documents\4th_Year_Bristol_University\Applied_data_science\Project\creditcard_data\Applied-Data-Science\creditcard.csv')


print(credit_card_data.columns)
corr=credit_card_data.corr()

plt.figure(1)
ax = sns.heatmap(
    corr, 
    vmin=-1, vmax=1, center=0,
    cmap=sns.diverging_palette(20, 220, n=200),
    square=False
)
ax.set_xticklabels(
    ax.get_xticklabels()
)
x=np.linspace(0,31,31)
pca=PCA(n_components=31)
model=pca.fit(corr)
components=pca.components_
explained_variance=pca.explained_variance_

plt.figure(2)
plt.plot(x,explained_variance,marker='o',linestyle='--')
plt.xlabel('Components')
plt.ylabel('Explained variance')
X_r=pca.transform(corr)
print('Correlation matrix',corr)
print('Components',components)
print('Explained variance',pca.explained_variance_)
print('Explained variance ratio',pca.explained_variance_ratio_)
print('Explained variance ratio cumulative sum',pca.explained_variance_ratio_.cumsum())

scaler=StandardScaler()
std=scaler.fit_transform(credit_card_data)
pca=PCA()
pca.fit(std)

plt.figure(3)
plt.plot(range(0,31),pca.explained_variance_ratio_.cumsum(),marker='o',linestyle='--')
plt.title('Explained variance by components')
plt.xlabel('Number of components')
plt.ylabel('Cumulative explained variance')
plt.grid()


plt.figure(4)
plt.scatter(credit_card_data.iloc[:,0],credit_card_data.iloc[:,29],)
pca=PCA(n_components=21)
pca.fit(std)
pca.transform(std)
scores_pca=pca.transform(std)
WCSS=[]
for i in range(1,21):
    kmeans_pca=KMeans(n_clusters=i,init='k-means++',random_state=42)
    kmeans_pca.fit(scores_pca)
    WCSS.append(kmeans_pca.inertia_)
plt.figure(5)
plt.plot(range(1,21),WCSS,marker='o',linestyle='--')
plt.xlabel('# Clusters')
plt.ylabel('WCSS')

kmeans_pca=KMeans(n_clusters=5,init='k-means++',random_state=42)
kmeans_pca.fit(scores_pca)
credit_card_data_pca_kmeans=pd.concat([credit_card_data.reset_index(drop=True),pd.DataFrame(scores_pca)],axis=1)
credit_card_data_pca_kmeans.columns.values[-3: ]=['Component 1','Component 2','Component 3']
credit_card_data_pca_kmeans['Segment k-means PCA']=kmeans_pca.labels_
credit_card_data_pca_kmeans['Segment']=credit_card_data_pca_kmeans['Segment k-means PCA'].map({0:'first',
1:'second',
2:'third',
3:'fourth'})
x_axis=credit_card_data_pca_kmeans['Component 2']
y_axis=credit_card_data_pca_kmeans['Component 1']

plt.figure(6)
sns.scatterplot(x_axis,y_axis,hue=credit_card_data_pca_kmeans['Segment'],palette=['g','r','c','m'])

plt.show()