# Import libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
import time

# Read in data and build matrices
train_df = pd.read_csv('data/train_potus_by_county.csv')
X = train_df[train_df.columns[:-1]].as_matrix()
y = train_df['Winner'].as_matrix()

log = open("performance.txt", "w")
log.write("We can test the performance of various untuned models using a 10-fold cross validation on the training data. Since we are classifying numerical features into two categories we explore the following models.\n")
log.write('Testing performance of machine learning models: Decision Tree, Kernel SVM, Random Forest, kNN...\n')

# Decision Tree
from sklearn import tree

log.write('\nRunning 10-fold cross validation with decision tree classifier.\n')
start = time.time()
clf = tree.DecisionTreeClassifier(random_state=0)
scores = cross_val_score(estimator=clf,
                X=X,
                y=y,
                n_jobs=-1,
                cv=10,
                verbose=1)
log.write('Scores: ' + str(scores) + '\n')
log.write('Time taken: %fs\n' % (time.time() - start))
log.write('The decision tree classifier results in an average accuracy of %f%% with 10-fold cross validation.\n' % (np.mean(scores)*100))

# SVM with RBF Kernel
from sklearn import svm

log.write('\nRunning 10-fold cross validation with kernel SVM classifier.\n')
start = time.time()
clf = svm.SVC(kernel='rbf', random_state=0)
scores = cross_val_score(estimator = clf,
                         X = X,
                         y = y,
                         n_jobs = -1,
                         cv = 10,
                         verbose = 1)
log.write('Scores: ' + str(scores) + '\n')
log.write('Time taken: %fs\n' % (time.time() - start))
log.write('The rbf support vector machine classifier results in an average accuracy of %f%% with 10-fold cross validation.\n' % (np.mean(scores)*100))

# Random Forest
from sklearn.ensemble import RandomForestClassifier

log.write('\nRunning 10-fold cross validation with random forest classifier.\n')
start = time.time()
clf = RandomForestClassifier(random_state=0)
scores = cross_val_score(estimator = clf,
                         X = X,
                         y = y,
                         n_jobs = -1,
                         cv = 10,
                         verbose = 1)
log.write('Scores: ' + str(scores) + '\n')
log.write('Time taken: %fs\n' % (time.time() - start))
log.write('The random forest classifier results in an average accuracy of %f%% with 10-fold cross validation.\n' % (np.mean(scores)*100))

# k-Nearest Neighbors
from sklearn.neighbors import KNeighborsClassifier

log.write('\nRunning 10-fold cross validation with kNN classifier.\n')
start = time.time()
clf = KNeighborsClassifier()
scores = cross_val_score(estimator = clf,
                         X = X,
                         y = y,
                         n_jobs = -1,
                         cv = 10,
                         verbose = 1)
log.write('Scores: ' + str(scores) + '\n')
log.write('Time taken: %fs\n' % (time.time() - start))
log.write('The kNN classifier results in an average accuracy of %f%% with 10-fold cross validation.\n' % (np.mean(scores)*100))

# We get the best results with random forest classifiers. Let's attempt to tune the hyperparameters.
from sklearn.model_selection import GridSearchCV

# Helper method to return results
def report(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            log.write("Model with rank: {0}\n".format(i))
            log.write("Mean validation score: {0:.3f} (std: {1:.3f})\n".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            log.write("Parameters: {0}\n".format(results['params'][candidate]))
            log.write("")

# Use grid search to fine tune parameters for a forest with small number of estimators

log.write('\nUsing grid search before feature engineering to find optimal hyperparameters.\n')

clf = RandomForestClassifier(n_estimators=20, random_state=0)
param_grid = {"max_depth": [3, None],
              "max_features": [1, 3, 10],
              "min_samples_split": [2, 3, 10],
              "min_samples_leaf": [1, 3, 10],
              "oob_score": [True, False],
              "criterion": ["gini", "entropy"]}

# run grid search
grid_search = GridSearchCV(clf, param_grid=param_grid)
start = time.time()
grid_search.fit(X, y)

log.write("GridSearchCV took %.2f seconds for %d candidate parameter settings.\n"
      % (time.time() - start, len(grid_search.cv_results_['params'])))
report(grid_search.cv_results_)

# Let's see if we can improve this performance with some feature engineering
#
# Random forest algorithms are invariant to linear transformations of our data, so we can use our domain knowledge
# to instead attempt to combine features to make them more relevant

# Population growth is a multiplier on total population
train_df['Population growth'] = train_df['Population growth'] * train_df['Total population']

# Percentage of people with bachelors degree or higher can be quantified
train_df['% BachelorsDeg or higher'] = train_df['Total population'] * train_df['% BachelorsDeg or higher']

# Unemployment rate can be quanitified to number unemployed
train_df['Unemployment rate'] = train_df['Unemployment rate'] * train_df['Total population']

# Household growth is a multiplier on the number of households
train_df['House hold growth'] = train_df['House hold growth'] * train_df['Total households']

# Per capita income growth is a multiplier on per capita income
train_df['Per capita income growth'] = train_df['Per capita income growth'] * train_df['Per capita income']

# Recreate matrices
X = train_df[train_df.columns[:-1]].as_matrix()
y = train_df['Winner'].as_matrix()

# Use grid search again to fine tune parameters on our new feature set

log.write('\nUsing grid search after feature engineering to find optimal hyperparameters.\n')

clf = RandomForestClassifier(n_estimators=20, random_state=0)
param_grid = {"max_depth": [3, None],
              "max_features": [1, 3, 10],
              "min_samples_split": [2, 3, 10],
              "min_samples_leaf": [1, 3, 10],
              "oob_score": [True, False],
              "criterion": ["gini", "entropy"]}

# run grid search
grid_search = GridSearchCV(clf, param_grid=param_grid)
start = time.time()
grid_search.fit(X, y)

log.write("GridSearchCV took %.2f seconds for %d candidate parameter settings.\n"
      % (time.time() - start, len(grid_search.cv_results_['params'])))
report(grid_search.cv_results_)

# Now that we know the optimum model and hyperparameters, we can train the model and save it
import pickle

log.write('\nTraining optimal model with tuned hyperparameters.\n')
start = time.time()
clf = RandomForestClassifier(n_estimators=1000,
                             min_samples_split=10,
                             min_samples_leaf=1,
                             max_features=3,
                             criterion='entropy',
                             oob_score=True,
                             max_depth=None,
                             verbose=1).fit(X, y)
with open('model.pickle', 'wb') as handle:
    pickle.dump(clf, handle, protocol=pickle.HIGHEST_PROTOCOL)
log.write('Time taken: %fs\n' % (time.time() - start))
log.write('Model trained and stored in file model.pkl\n')

log.close()

