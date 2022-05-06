"""
-- cite --
    https://www.kaggle.com/code/janiobachmann/credit-fraud-dealing-with-imbalanced-datasets/notebook
"""

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA, TruncatedSVD
import matplotlib.patches as mpatches
import time

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
import collections

# Other Libraries
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import learning_curve
from sklearn.pipeline import make_pipeline
from imblearn.pipeline import make_pipeline as imbalanced_make_pipeline
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import NearMiss
from imblearn.metrics import classification_report_imbalanced
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score, classification_report
from collections import Counter
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler, RobustScaler
import warnings
warnings.filterwarnings("ignore")

def plot(estimator, X, y, ylim=None, cv=None, n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    ax1 = plt.subplot()
    if ylim is not None:
        plt.ylim(*ylim)

    #Estimator
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    ax1.plot(train_sizes, train_scores_mean, 'o-', color="#ff9124",
             label="Training score")
    ax1.plot(train_sizes, test_scores_mean, 'o-', color="#2492ff",
             label="Cross-validation score")
    ax1.set_title("Logistic Regression Learning Curve", fontsize=14)
    ax1.set_xlabel('Training size (m)')
    ax1.set_ylabel('Score')
    ax1.grid(True)
    ax1.legend(loc="best")
    return plt


df = pd.read_csv('creditcard.csv')
# print(df.head())

# RobustScaler is less prone to outliers.
std_scaler = StandardScaler()
rob_scaler = RobustScaler()

df['scaled_amount'] = rob_scaler.fit_transform(df['Amount'].values.reshape(-1,1))
df['scaled_time'] = rob_scaler.fit_transform(df['Time'].values.reshape(-1,1))

df.drop(['Time','Amount'], axis=1, inplace=True)

scaled_amount = df['scaled_amount']
scaled_time = df['scaled_time']

df.drop(['scaled_amount', 'scaled_time'], axis=1, inplace=True)
df.insert(0, 'scaled_amount', scaled_amount)
df.insert(1, 'scaled_time', scaled_time)

# Since our classes are highly skewed we should make them equivalent in order to have a normal distribution of the classes.
# We shuffle the data before creating the subsamples
df = df.sample(frac=1)

# amount of fraud classes 492 rows.
fraud_df = df.loc[df['Class'] == 1]
non_fraud_df = df.loc[df['Class'] == 0][:492]

normal_distributed_df = pd.concat([fraud_df, non_fraud_df])

# Shuffle dataframe rows
new_df = normal_distributed_df.sample(frac=1, random_state=42)

# Undersampling before cross validating
X = new_df.drop('Class', axis=1)
y = new_df['Class']

# Our data is already scaled we should split our training and test sets
# This is explicitly used for undersampling.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Turn the values into an array for feeding the classification algorithms.
X_train = X_train.values
X_test = X_test.values
y_train = y_train.values
y_test = y_test.values

# Classifiers
classifiers = {
    "LogisiticRegression": LogisticRegression(),
    "KNearest": KNeighborsClassifier(),
    "Support Vector Classifier": SVC(),
    "DecisionTreeClassifier": DecisionTreeClassifier(),
    "MLPClassifier": MLPClassifier(),
    "AdaBoostClassifier": AdaBoostClassifier(),
    "RandomForestClassifier": RandomForestClassifier(),
    "GradientBoostingClassifier": GradientBoostingClassifier()
}

for key, classifier in classifiers.items():
    classifier.fit(X_train, y_train)
    training_score = cross_val_score(classifier, X_train, y_train, cv=5)
    # print("Classifiers: ", classifier.__class__.__name__, "Has a training score of", round(training_score.mean(), 2) * 100, "% accuracy score")

# ----------------------------------------------------------------------------------------------------------
# Logistic Regression
log_reg_params = {"penalty": ['l1', 'l2'], 'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}

grid_log_reg = GridSearchCV(LogisticRegression(), log_reg_params)
grid_log_reg.fit(X_train, y_train)
# Logistic regression best estimator
log_reg = grid_log_reg.best_estimator_
print("log_reg:", log_reg)
# ----------------------------------------------------------------------------------------------------------
# Knears
knears_params = {"n_neighbors": list(range(2,5,1)), 'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']}

grid_knears = GridSearchCV(KNeighborsClassifier(), knears_params)
grid_knears.fit(X_train, y_train)
# KNears best estimator
knears_neighbors = grid_knears.best_estimator_
print("knears_neighbors:", knears_neighbors)
# ----------------------------------------------------------------------------------------------------------
# Support Vector Classifier
svc_params = {'C': [0.5, 0.7, 0.9, 1], 'kernel': ['rbf', 'poly', 'sigmoid', 'linear']}

grid_svc = GridSearchCV(SVC(), svc_params)
grid_svc.fit(X_train, y_train)
# SVC best estimator
svc = grid_svc.best_estimator_
print("svc:", svc)
# ----------------------------------------------------------------------------------------------------------
# DecisionTree Classifier
tree_params = {"criterion": ["gini", "entropy"], "max_depth": list(range(2,4,1)),
              "min_samples_leaf": list(range(5,7,1))}

grid_tree = GridSearchCV(DecisionTreeClassifier(), tree_params)
grid_tree.fit(X_train, y_train)
# tree best estimator
tree_clf = grid_tree.best_estimator_
print("tree_clf:", tree_clf)
# ----------------------------------------------------------------------------------------------------------
# MLP
mlp_params = {'activation': ['logistic'], 'solver': ['lbfgs', 'sgd', 'adam']} #['identity', 'logistic', 'tanh', 'relu']

grid_mlp = GridSearchCV(MLPClassifier(), mlp_params)
grid_mlp.fit(X_train, y_train)
# MLP best estimator
mlp = grid_mlp.best_estimator_
print("mlp:", mlp)
# ----------------------------------------------------------------------------------------------------------
# AdaBoostClassifier
adaboost_params = {'n_estimators': list(range(10,100,10))}

grid_adaboost = GridSearchCV(AdaBoostClassifier(), adaboost_params)
grid_adaboost.fit(X_train, y_train)
# AdaBoost best estimator
ada_boost = grid_adaboost.best_estimator_
print("ada_boost:", ada_boost)
# ----------------------------------------------------------------------------------------------------------
# RandomForestClassifier
random_forest_params = {'n_estimators': list(range(10,100,10)), 'criterion': ['gini', 'entropy']}

grid_random_forest = GridSearchCV(RandomForestClassifier(), random_forest_params)
grid_random_forest.fit(X_train, y_train)
# Random Forest best estimator
random_forest = grid_random_forest.best_estimator_
print("random_forest:", random_forest)
# ----------------------------------------------------------------------------------------------------------
# GradientBoostingClassifier
# gradient_boost_params = {'n_estimators': list(range(10,100,10)), 'criterion': ['friedman_mse', 'squared_error', 'mse', 'mae'], 'max_depth': list(range(2,4,1))}
#
# grid_gradient_boost = GridSearchCV(GradientBoostingClassifier(), gradient_boost_params)
# grid_gradient_boost.fit(X_train, y_train)
# # GradientBoost best estimator
# gradient_boost = grid_gradient_boost.best_estimator_
# print("gradient_boost:", gradient_boost)

log_reg_score = cross_val_score(log_reg, X_test, y_test, cv=5)
print('Logistic Regression Cross Validation Score: ', round(log_reg_score.mean() * 100, 2).astype(str) + '%')

knears_score = cross_val_score(knears_neighbors, X_test, y_test, cv=5)
print('Knears Neighbors Cross Validation Score', round(knears_score.mean() * 100, 2).astype(str) + '%')

svc_score = cross_val_score(svc, X_test, y_test, cv=5)
print('Support Vector Classifier Cross Validation Score', round(svc_score.mean() * 100, 2).astype(str) + '%')

tree_score = cross_val_score(tree_clf, X_test, y_test, cv=5)
print('DecisionTree Classifier Cross Validation Score', round(tree_score.mean() * 100, 2).astype(str) + '%')

mlp_score = cross_val_score(mlp, X_test, y_test, cv=5)
print('MLP Classifier Cross Validation Score', round(mlp_score.mean() * 100, 2).astype(str) + '%')

ada_boost_score = cross_val_score(ada_boost, X_test, y_test, cv=5)
print('AdaBoostClassifier Cross Validation Score', round(ada_boost_score.mean() * 100, 2).astype(str) + '%')

random_forest_score = cross_val_score(random_forest, X_test, y_test, cv=5)
print('RandomForestClassifier Cross Validation Score', round(random_forest_score.mean() * 100, 2).astype(str) + '%')

# gradient_boost_score = cross_val_score(gradient_boost, X_test, y_test, cv=5)
# print('GradientBoostingClassifier Cross Validation Score', round(gradient_boost_score.mean() * 100, 2).astype(str) + '%')


cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=42)

plt = plot(log_reg, X_train, y_train, (0.87, 1.01), cv=cv, n_jobs=4)
plt.show()
