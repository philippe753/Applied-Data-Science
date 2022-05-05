from cProfile import label
from numpy import unique

from sklearn import metrics

from sklearn.manifold import TSNE
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

credit_card_data=pd.read_csv(r'C:\Users\tmara\Documents\4th_Year_Bristol_University\Applied_data_science\Project\creditcard_data\Applied-Data-Science\balanced_data.csv')
full_credit_card_data=pd.read_csv(r'C:\Users\tmara\Documents\4th_Year_Bristol_University\Applied_data_science\Project\creditcard_data\Applied-Data-Science\creditcard.csv')
imbalanced_sample=pd.DataFrame(full_credit_card_data).sample(n=5000,random_state=1)
credit_card_data= credit_card_data.drop("Unnamed: 0",axis=1)
covariance_matrix=np.cov(credit_card_data)
print(covariance_matrix)
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

def tsne_function(dataset,learning_rate=50,perplexity=20,n_components=2):
    df=pd.DataFrame([])
    df['Original']=dataset['Class']
    true_labels=np.array(df['Original'])
    tsne_dataframe=pd.DataFrame([])
    tsne_dataframe['Class']=true_labels
    dataset= dataset.drop("Class",axis=1)
    dataset= dataset.drop("Time",axis=1)
    dataset= dataset.drop("Amount",axis=1)
    dataset=dataset.drop(['V8','V15','V13','V23', 'V24', 'V25', 'V26', 'V27', 'V28'],axis=1)
    print(dataset.head(5))

    tsne_dataframe['Class']=tsne_dataframe['Class'].replace(0,'Non-Fraud')
    tsne_dataframe['Class']=tsne_dataframe['Class'].replace(1,'Fraud')
    model=TSNE(learning_rate=learning_rate,perplexity=perplexity,n_components=n_components)
    tsne_features=model.fit_transform(dataset)
    tsne_labels=tsne_features.get_params()
    print('TSNE labels',tsne_labels)
    print(tsne_features[1:4,:])
    tsne_dataframe['x']=tsne_features[:,0]
    tsne_dataframe['y']=tsne_features[:,1]

    count_dataset=counter(true_labels)
    print('Number of frauds in data set',count_dataset)
    
    plt.figure()
    plt.title('t-SNE')
    sns.scatterplot(x='x',y='y',hue='Class',data=tsne_dataframe)
    plt.show()

tsne_function(upsampling,perplexity=50)