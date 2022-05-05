from cProfile import label
from cv2 import kmeans
from numpy import unique

from sklearn import metrics

from sklearn.cluster import KMeans
import pandas as pd

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

import numpy as np
from sklearn.preprocessing import StandardScaler

import sklearn.metrics.cluster
import seaborn as sns

from sklearn.model_selection import train_test_split

credit_card_data=pd.read_csv(r'C:\Users\tmara\Documents\4th_Year_Bristol_University\Applied_data_science\Project\creditcard_data\Applied-Data-Science\balanced_data.csv')

full_credit_card_data=pd.read_csv(r'C:\Users\tmara\Documents\4th_Year_Bristol_University\Applied_data_science\Project\creditcard_data\Applied-Data-Science\creditcard.csv')
imbalanced_sample=pd.DataFrame(full_credit_card_data).sample(n=5000,random_state=20)

credit_card_data= credit_card_data.drop("Unnamed: 0",axis=1)
def counter(array):
      count=0
      for i in array:
            if i==1:
                  count+=1
            if i==0:
                  count+=0
      return count

def kmeans_function(dataset):
      df=pd.DataFrame([])
      df['Original']=dataset['Class']
      true_labels=np.array(df['Original'])
      print('True labels',true_labels)

      dataset= dataset.drop("Class",axis=1)
      dataset= dataset.drop("Time",axis=1)
      dataset= dataset.drop("Amount",axis=1)
      dataset=dataset.drop(['V8','V15','V13','V23', 'V24', 'V25', 'V26', 'V27', 'V28'],axis=1)
      #print(dataset.head(5))

      # test_size: what proportion of original data is used for test set
      #---------------------------------------------------------------------PCA--------------------------------------------------------------------------
      scaler=StandardScaler()
      std=scaler.fit_transform(dataset)
      pca=PCA()

      #print(pca.explained_variance_ratio_)

      pca=PCA(n_components=4)
      X_pca=pca.fit_transform(std)
      X_pca=np.vstack((X_pca[true_labels == 0], X_pca[true_labels == 1]))
      kmeansc = KMeans(n_clusters=2).fit(X_pca)

      labels = kmeansc.labels_
      prediction=kmeansc.fit_predict(X_pca)

      # Number of clusters in labels, ignoring noise if present.
      n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
      

      # Plot result
      
      # Black removed and is used for noise instead.
      plt.figure(1)

      plt.scatter(X_pca[true_labels==1, 0],
            X_pca[true_labels==1, 1],

            c=prediction[true_labels==1])
      plt.scatter(X_pca[true_labels==0, 0],
            X_pca[true_labels==0, 1],

            c=prediction[true_labels==0])
      plt.title('K-means clustering')

      plt.xlabel('Component 1')
      plt.ylabel('Component 2')
      plt.legend()

      count_prediction=counter(prediction)
      count_dataset=counter(true_labels)
      print('Number of frauds in data set',count_dataset)
      print('Number of frauds in prediction',count_prediction)
      labels=np.where(labels==-1,1,labels)

      print('Estimated number of clusters: %d' % n_clusters_)
      print("Homogeneity: %0.3f" % metrics.homogeneity_score(true_labels, labels))
      print("Completeness: %0.3f" % metrics.completeness_score(true_labels, labels))
      print("V-measure: %0.3f" % metrics.v_measure_score(true_labels, labels))
      print("Adjusted Rand Index: %0.3f"
            % metrics.adjusted_rand_score(true_labels, labels))
      print("Adjusted Mutual Information: %0.3f"
            % metrics.adjusted_mutual_info_score(true_labels, labels))
      print("Silhouette Coefficient: %0.3f"
            % metrics.silhouette_score(X_pca, labels))
      # retrieve unique clusters
      print("Calinski-Harabasz Index: %0.3f"
            % sklearn.metrics.calinski_harabasz_score(X_pca, labels))
      print(metrics.accuracy_score(np.array(df['Original']),prediction))
      print(metrics.confusion_matrix(np.array(df['Original']),prediction))
      plt.show()

kmeans_function(imbalanced_sample)