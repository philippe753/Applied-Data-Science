from turtle import color
import numpy as npgit
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import statistics as stats
from scipy.stats import norm
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, SpectralClustering
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
credit_card_data=pd.read_csv(r'C:\Users\tmara\Documents\4th_Year_Bristol_University\Applied_data_science\Project\creditcard_data\Applied-Data-Science\balanced_data.csv')


credit_card_data= credit_card_data.drop("Unnamed: 0",axis=1)


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
plt.title('Correlation Matrix')
credit_card_data= credit_card_data.drop("Class",axis=1)
credit_card_data= credit_card_data.drop("Time",axis=1)
credit_card_data= credit_card_data.drop("Amount",axis=1)
credit_card_data=credit_card_data.drop(['V8','V15','V13','V23', 'V24', 'V25', 'V26', 'V27', 'V28'],axis=1)
scaler=StandardScaler()
std=scaler.fit_transform(credit_card_data)
pca=PCA()
pca.fit(std)
print(pca.explained_variance_ratio_)
plt.figure(3)
plt.plot(range(1,20),pca.explained_variance_ratio_.cumsum(),marker='o',linestyle='--',label='Cumulative variance')
plt.title('Explained variance by components')
plt.plot(4,0.8457,marker='x',color='r',markersize=10)
plt.plot(range(1,20),pca.explained_variance_ratio_,marker='o',linestyle='--',label='Variance')
plt.xlabel('Number of components')
plt.ylabel('Proportion of variance')
plt.legend()
plt.grid()


pca=PCA(n_components=4)
pca.fit(std)
pca.transform(std)
scores_pca=pca.transform(std)
WCSS=[]
for i in range(1,5):
    kmeans_pca=KMeans(n_clusters=i,init='k-means++',random_state=42)
    kmeans_pca.fit(scores_pca)
    WCSS.append(kmeans_pca.inertia_)
plt.figure(5)
plt.plot(range(1,5),WCSS,marker='o',linestyle='--')
plt.xlabel('# Clusters') 
plt.ylabel('WCSS')
plt.title('Elbow method')
plt.plot(2,7310,marker='x',color='r',markersize=10)
plt.grid()
kmeans_pca=KMeans(n_clusters=2,init='k-means++',random_state=2)
kmeans_pca.fit(scores_pca)
credit_card_data_pca_kmeans=pd.concat([credit_card_data.reset_index(drop=True),pd.DataFrame(scores_pca)],axis=1)
credit_card_data_pca_kmeans.columns.values[-4: ]=['Component 1','Component 2','Component 3','Component 4']
print('Components',credit_card_data_pca_kmeans.columns)
credit_card_data_pca_kmeans['Segment k-means PCA']=kmeans_pca.labels_
credit_card_data_pca_kmeans['Segment']=credit_card_data_pca_kmeans['Segment k-means PCA'].map({0:'first',
1:'second'})
x_axis=credit_card_data_pca_kmeans['Component 1']
y_axis=credit_card_data_pca_kmeans['Component 2']
z_axis=credit_card_data_pca_kmeans['Component 3']

plt.figure(6)
sns.scatterplot(x_axis,y_axis,hue=credit_card_data_pca_kmeans['Segment'],palette=['g','r'])
plt.title('PCA and K-Means clustering')
plt.grid()

fig = plt.figure(7)
ax = fig.add_subplot(111, projection='3d')

# assign x,y,z coordinates from PC1, PC2 & PC3
xs = x_axis
ys = y_axis
zs = z_axis

# initialize scatter plot and label axes
ax.scatter(xs, ys, zs, alpha=0.75, c='blue')
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_zlabel('PC3')
plt.show()