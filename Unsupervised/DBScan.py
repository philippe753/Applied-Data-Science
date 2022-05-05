from numpy import unique

from sklearn import metrics

from sklearn.cluster import DBSCAN

import pandas as pd

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

import numpy as np
from sklearn.preprocessing import StandardScaler

import sklearn.metrics.cluster
import seaborn as sns

from sklearn.model_selection import train_test_split

downsampling=pd.read_csv(r'C:\Users\tmara\Documents\4th_Year_Bristol_University\Applied_data_science\Project\creditcard_data\Applied-Data-Science\balanced_data.csv')
downsampling= downsampling.drop("Unnamed: 0",axis=1)
rawdataset=pd.read_csv(r'C:\Users\tmara\Documents\4th_Year_Bristol_University\Applied_data_science\Project\creditcard_data\Applied-Data-Science\creditcard.csv')

upsampling=pd.read_csv(r'C:\Users\tmara\Documents\4th_Year_Bristol_University\Applied_data_science\Project\creditcard_data\Applied-Data-Science\final_oversampled_data_smote.csv')
upsampling=upsampling.drop("Unnamed: 0",axis=1)
upsampling=upsampling.drop("Unnamed: 0.1",axis=1)
def counter(array):
      count=0
      for i in array:
            if i==1:
                  count+=1
            if i==0:
                  count+=0
      return count
def balanced_subsample(y, size=None):

    subsample = []

    if size is None:
        n_smp = y.value_counts().min()
    else:
        n_smp = int(size / len(y.value_counts().index))

    for label in y.value_counts().index:
        samples = y[y == label].index.values
        index_range = range(samples.shape[0])
        indexes = np.random.choice(index_range, size=n_smp, replace=False)
        subsample += samples[indexes].tolist()

    return subsample
balanced_sub=balanced_subsample(upsampling['Class'],size=60000)
balanced_sub=upsampling.iloc[balanced_sub]


def dbscan_function(dataset,epsilon=1.4,min_samples=20000):
      dbscan_dataframe=pd.DataFrame([])
      dbscan_dataframe['True labels']=dataset['Class']
      true_labels=np.array(dbscan_dataframe['True labels'])
      dataset= dataset.drop("Class",axis=1)
      dataset= dataset.drop("Time",axis=1)
      dataset= dataset.drop("Amount",axis=1)
      dataset=dataset.drop(['V8','V15','V13','V23', 'V24', 'V25', 'V26', 'V27', 'V28'],axis=1)
      #---------------------------------------------------------------------PCA--------------------------------------------------------------------------
      scaler=StandardScaler()
      std=scaler.fit_transform(dataset)
      pca=PCA()
      pca.fit(std)
      pca=PCA(n_components=4)
      X_pca=pca.fit_transform(std)
      print(np.shape(X_pca))
      X_pca_dataframe=pd.DataFrame(X_pca)
      print(X_pca_dataframe)
      dbscan_dataframe['Component 1']=X_pca_dataframe[0]
      dbscan_dataframe['Component 2']=X_pca_dataframe[1]
      dbscan_dataframe['Component 3']=X_pca_dataframe[2]
      dbscan_dataframe['Component 4']=X_pca_dataframe[3]
      model = DBSCAN(eps=epsilon, min_samples=min_samples).fit(X_pca)
      dbscan_feature=model.labels_
      print('model results',dbscan_feature)
      
      core_samples_mask = np.zeros_like(model.labels_, dtype=bool)
      core_samples_mask[model.core_sample_indices_] = True
      n_clusters_ = len(set(dbscan_feature)) - (1 if 1 in dbscan_feature else 0)
      n_noise_ = list(dbscan_feature).count(-1)
      print('Labels',dbscan_feature)
      print('True labels',true_labels)
      print('Estimated number of clusters: %d' % n_clusters_)
      print('Estimated number of noise points: %d' % n_noise_)
      print("Homogeneity: %0.3f" % metrics.homogeneity_score(true_labels, dbscan_feature))
      print("Completeness: %0.3f" % metrics.completeness_score(true_labels, dbscan_feature))
      print("V-measure: %0.3f" % metrics.v_measure_score(true_labels, dbscan_feature))
      print("Adjusted Rand Index: %0.3f"
            % metrics.adjusted_rand_score(true_labels, dbscan_feature))
      print("Adjusted Mutual Information: %0.3f"
            % metrics.adjusted_mutual_info_score(true_labels, dbscan_feature))
      print("Silhouette Coefficient: %0.3f"
            % metrics.silhouette_score(X_pca, dbscan_feature))
      # retrieve unique clusters
      print("Calinski-Harabasz Index: %0.3f"
            % sklearn.metrics.calinski_harabasz_score(X_pca, dbscan_feature))
      dbscan_feature=np.where(dbscan_feature==-1,1,dbscan_feature)
      dbscan_dataframe['results']=dbscan_feature
      count_prediction=counter(dbscan_feature)
      count_dataset=counter(true_labels)
      print('Number of frauds in data set',count_dataset)
      print('Number of frauds in prediction',count_prediction)

      dbscan_dataframe['Class']=dbscan_dataframe['results'].replace(0,'Non-Fraud')
      dbscan_dataframe['Class']=dbscan_dataframe['Class'].replace(1,'Fraud')
      print('Dataframe',dbscan_dataframe)

      fig = plt.figure()
      ax = fig.add_subplot()
      ax.scatter(X_pca[dbscan_dataframe['Class']=='Fraud', 0],
            X_pca[dbscan_dataframe['Class']=='Fraud', 1],

            c='lightgreen',
            label='Fraud')
      ax.scatter(X_pca[dbscan_dataframe['Class']=='Non-Fraud', 0],
            X_pca[dbscan_dataframe['Class']=='Non-Fraud', 1],


            c='orange',
            label='Non-fraud')
      print('Accuracy score',metrics.accuracy_score(np.array(dbscan_dataframe['True labels']),np.array(dbscan_dataframe['results'])))
      print('Confusion matrix',metrics.confusion_matrix(np.array(dbscan_dataframe['True labels']),np.array(dbscan_dataframe['results'])))

      """plt.figure(7)
      print('Unique classes',unique(dbscan_dataframe['results']))
      sns.scatterplot(x=dbscan_dataframe['Component 1'],y=dbscan_dataframe['Component 2'],
            hue='Class',
            data=dbscan_dataframe)"""

      plt.title('DBScan clustering')

      plt.xlabel('Component 1')
      plt.ylabel('Component 2')
      plt.legend()

      
      plt.show()

dbscan_function(balanced_sub)