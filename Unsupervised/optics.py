from numpy import unique

from sklearn import metrics

from sklearn.cluster import OPTICS

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
imbalanced_sample=pd.DataFrame(full_credit_card_data).sample(n=10000,random_state=10)


credit_card_data= credit_card_data.drop("Unnamed: 0",axis=1)


#print(credit_card_data.head(5))

# test_size: what proportion of original data is used for test set

def counter(array):
      count=0
      for i in array:
            if i==1:
                  count+=1
            if i==0:
                  count+=0
      return count

#---------------------------------------------------------------------PCA--------------------------------------------------------------------------
def optics_function(dataset,min_samples=110):
      df=pd.DataFrame([])
      df['Original']=dataset['Class']
      true_labels=np.array(df['Original'])
      print('True labels',true_labels)

      dataset= dataset.drop("Class",axis=1)
      dataset= dataset.drop("Time",axis=1)
      dataset= dataset.drop("Amount",axis=1)
      dataset=dataset.drop(['V8','V15','V13','V23', 'V24', 'V25', 'V26', 'V27', 'V28'],axis=1)
      scaler=StandardScaler()
      std=scaler.fit_transform(dataset)
      pca=PCA()
      pca.fit(std)
      #print(pca.explained_variance_ratio_)

      pca=PCA(n_components=4)
      X_pca=pca.fit_transform(std)
      print('X_pca',X_pca)
      db = OPTICS(min_samples=min_samples).fit(X_pca)

      labels = db.labels_
      print(X_pca)
      # Number of clusters in labels, ignoring noise if present.
      n_clusters_ = len(set(labels)) - (1 if 1 in labels else 0)
      
      print(labels)
      
      # Plot result
      
      # Black removed and is used for noise instead.
      unique_labels = set(labels)
      colors = ['y', 'b', 'g', 'r']
      print(colors)
      labels=np.where(labels==-1,1,labels)

      fig = plt.figure()
      ax = fig.add_subplot(projection='3d')
      ax.scatter(X_pca[labels==1, 0],
            X_pca[labels==1, 1],
            X_pca[labels==1, 2],
            c='lightgreen',
            label='Fraud')
      ax.scatter(X_pca[labels==0, 0],
            X_pca[labels==0, 1],
            X_pca[labels==0, 2],

            c='orange',
            label='Non-fraud')

      #fig.set_title('OPTICS clustering')

      ax.set_xlabel('Component 1')
      ax.set_ylabel('Component 2')
      ax.set_zlabel('Component 3')
      plt.legend()
      count_prediction=counter(labels)
      count_dataset=counter(true_labels)
      print('Number of frauds in data set',count_dataset)
      print('Number of frauds in prediction',count_prediction)
      
      print('Labels',labels)
      print('Labels true',true_labels)
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

      print(metrics.accuracy_score(np.array(df['Original']),labels))
      print(metrics.confusion_matrix(np.array(df['Original']),labels))
      plt.show()

optics_function(credit_card_data)