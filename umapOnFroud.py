import umap.umap_ as umap
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# import balanced dataset:
# df= pd.read_csv('creditcard.csv')
df_balanced = pd.read_csv('Datasets/balanced_data.csv')

V_colums = df_balanced.columns[2:30]
df_Vs = df_balanced[V_colums]  # pd datatrame of only the Vs
print(df_balanced.columns[1])


reducer = umap.UMAP(random_state=42)
embedding = reducer.fit_transform(df_Vs)
print(embedding.shape)

plt.scatter(
    embedding[:, 0],
    embedding[:, 1],
    c=df_balanced.Class)
plt.gca().set_aspect('equal', 'datalim')
plt.colorbar(boundaries=np.arange(3)-0.5).set_ticks(np.arange(2))
plt.title('UMAP projection using all info', fontsize=24)
plt.show()
# embedding = reducer.transform(df_Vs)
# reducer3d = umap.UMAP(random_state=42, n_components=3)
# reducer3d.fit(df_Vs)
# embedding3d = reducer.transform(df_Vs)
