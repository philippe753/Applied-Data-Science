from cv2 import threshold
from numpy import unique

from sklearn import metrics

from sklearn.cluster import Birch

import pandas as pd

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

import numpy as np
from sklearn.preprocessing import StandardScaler

import sklearn.metrics.cluster
import seaborn as sns

from sklearn.model_selection import train_test_split

dataset=pd.read_csv(r'C:\Users\tmara\Documents\4th_Year_Bristol_University\Applied_data_science\Project\creditcard_data\Applied-Data-Science\balanced_data.csv')
dataset= dataset.drop("Unnamed: 0",axis=1)
full_credit_card_data=pd.read_csv(r'C:\Users\tmara\Documents\4th_Year_Bristol_University\Applied_data_science\Project\creditcard_data\Applied-Data-Science\creditcard.csv')
imbalanced_sample=pd.DataFrame(full_credit_card_data).sample(n=5000,random_state=150)
def counter(array):
      count=0
      for i in array:
            if i==1:
                  count+=1
            if i==0:
                  count+=0
      return count

def birch_function(dataset,threshold=0.05,n_clusters=2):
      birch_dataframe=pd.DataFrame([])
      birch_dataframe['True labels']=dataset['Class']
      true_labels=np.array(birch_dataframe['True labels'])
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
      birch_dataframe['Component 1']=X_pca_dataframe[0]
      birch_dataframe['Component 2']=X_pca_dataframe[1]
      birch_dataframe['Component 3']=X_pca_dataframe[2]
      birch_dataframe['Component 4']=X_pca_dataframe[3]
      model = Birch(threshold=threshold, n_clusters=n_clusters)
      model.fit(X_pca)
      y_hat=model.predict(X_pca)
      


      birch_prediction=np.where(y_hat==-1,1,y_hat)

      n_clusters_ = len(set(birch_prediction)) - (1 if 1 in birch_prediction else 0)
      n_noise_ = list(birch_prediction).count(-1)
      print('Labels',birch_prediction)
      print('True labels',true_labels)
      print('Estimated number of clusters: %d' % n_clusters_)
      print('Estimated number of noise points: %d' % n_noise_)
      print("Homogeneity: %0.3f" % metrics.homogeneity_score(true_labels, birch_prediction))
      print("Completeness: %0.3f" % metrics.completeness_score(true_labels, birch_prediction))
      print("V-measure: %0.3f" % metrics.v_measure_score(true_labels, birch_prediction))
      print("Adjusted Rand Index: %0.3f"
            % metrics.adjusted_rand_score(true_labels, birch_prediction))
      print("Adjusted Mutual Information: %0.3f"
            % metrics.adjusted_mutual_info_score(true_labels, birch_prediction))
      print("Silhouette Coefficient: %0.3f"
            % metrics.silhouette_score(X_pca, birch_prediction))
      # retrieve unique clusters
      print("Calinski-Harabasz Index: %0.3f"
            % sklearn.metrics.calinski_harabasz_score(X_pca, birch_prediction))

      birch_dataframe['results']=birch_prediction
      count_prediction=counter(birch_prediction)
      count_dataset=counter(true_labels)
      print('Number of frauds in data set',count_dataset)
      print('Number of frauds in prediction',count_prediction)

      birch_dataframe['Class']=birch_dataframe['results'].replace(0,'Non-Fraud')
      birch_dataframe['Class']=birch_dataframe['Class'].replace(1,'Fraud')
      print('Dataframe',birch_dataframe)
      plt.figure(7)

      sns.scatterplot(birch_dataframe['Component 3'],birch_dataframe['Component 4'],
            hue='Class',
            data=birch_dataframe)

      plt.title('DBScan clustering')

      plt.xlabel('Component 1')
      plt.ylabel('Component 2')
      plt.legend()

      print('Accuracy score',metrics.accuracy_score(np.array(birch_dataframe['True labels']),np.array(birch_dataframe['results'])))
      print('Confusion matrix',metrics.confusion_matrix(np.array(birch_dataframe['True labels']),np.array(birch_dataframe['results'])))
      plt.show()

birch_function(dataset)