#Modules

import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io

from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA
from sklearn.metrics import log_loss

from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, recall_score, precision_score, \
    f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline
from sklearn import metrics
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

from sklearn.datasets import make_blobs  # used for creating artifical datasets
% matplotlib
inline

'''Import Files'''

from google.colab import files

uploaded = files.upload()

for fn in uploaded.keys():
    print('User uploaded file "{name}" with length {length} bytes'.format(
        name=fn, length=len(uploaded[fn])))

from google.colab import files

uploaded = files.upload()

for fn in uploaded.keys():
    print('User uploaded file "{name}" with length {length} bytes'.format(
        name=fn, length=len(uploaded[fn])))

'''Files into pandas'''

dfbirths = pd.read_csv("TomBirths.csv")
dfbirths.head()

'''Overveiw of data'''

sns.pairplot(data=df, hue="AverageDeaths")

'''Formatting Data'''

# Choosing the data, this will only work for current data as it uses the last column in the diabeties data set as y
X = df.iloc[:, :-1]
y = df.Outcome
# Other ways
# X = df.drop('Outcome', axis=1)
# y = df.Outcome

'''Test train split'''
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Used for either missing data, or turning categorical data (male, female) into 1's and zero's 

# ('imputer', SimpleImputer(strategy='most_frequent')) #filling missing values

# (‘onehot', OneHotEncoder(handle_unknown='ignore'))    #convert categorical

'''Processing data through mall Machine Learning Algorithims'''

Dict_of_Accuracys = {}
Dict_of_Accuracys.clear()  # used for emptying Dict
Dict_of_Accuracys = {}

# Pipelines are API's that configured to perform preprocessing, in this instance it creates a standard scaler,
# goes through PCA(Principal component analysis) removing any outliers https://en.wikipedia.org/wiki/Principal_component_analysis
# finally the data is out through the algorithim you want

pipeline_lr = Pipeline([('scalar1', StandardScaler()),
                        ('pca1', PCA(n_components=2)),
                        ('lr_classifier', LogisticRegression())])
pipeline_dt = Pipeline([('scalar2', StandardScaler()),
                        ('pca2', PCA(n_components=2)),
                        ('dt_classifier', DecisionTreeClassifier())])
pipeline_svm = Pipeline([('scalar3', StandardScaler()),
                         ('pca3', PCA(n_components=2)),
                         ('clf', svm.SVC())])
pipeline_knn = Pipeline([('scalar4', StandardScaler()),
                         ('pca4', PCA(n_components=2)),
                         ('knn_classifier', KNeighborsClassifier())])
pipeline_rf = Pipeline([('scalar4', StandardScaler()),
                        ('pca4', PCA(n_components=2)),
                        ('knn_classifier', RandomForestClassifier())])
pipelines = [pipeline_lr, pipeline_dt, pipeline_svm, pipeline_knn, pipeline_rf]
pipe_dict = {0: 'Logistic Regression', 1: 'Decision Tree', 2: 'Support Vector Machine', 3: 'K Nearest Neighbor',
             4: 'Random Forest Classifier'}
for pipe in pipelines:
    pipe.fit(X_train, y_train)
for i, model in enumerate(pipelines):
    Dict_of_Accuracys[i] = model.score(X_test, y_test)
    print("{} Test Accuracy:{}".format(pipe_dict[i], model.score(X_test, y_test)))

print(Dict_of_Accuracys)

all_values = Dict_of_Accuracys.values()

max_value = max(all_values)

''' Seleting the best value and then printing the scores and confusion matrix '''

# After finding the best value and it's accompaningy alogrithim.
# It then puts that algorithim through tuning going through several different parameters, the funtion used is called grid search

# There is a difference between pipeline and make_pipeline but I am not sure what it is

if max_value == 0:
    clf = LogisticRegression()
    grid_values = {'penalty': ['l1', 'l2'], 'C': [0.001, .009, 0.01, .09, 1, 5, 10, 25]}
    grid_clf_acc = GridSearchCV(clf, param_grid=grid_values, scoring='recall')
    grid_clf_acc.fit(X_train, y_train)

    # Predict values based on new parameters
    y_pred_acc = grid_clf_acc.predict(X_test)

    # New Model Evaluation metrics
    print('Accuracy Score : ' + str(accuracy_score(y_test, y_pred_acc)))
    print('Precision Score : ' + str(precision_score(y_test, y_pred_acc)))
    print('Recall Score : ' + str(recall_score(y_test, y_pred_acc)))
    print('F1 Score : ' + str(f1_score(y_test, y_pred_acc)))

    # Logistic Regression (Grid Search) Confusion matrix
    confusion_matrix(y_test, y_pred_acc)




elif max_value == 1:
    # StandardScaler is used to remove the outliners and scale the data by making the mean of the data 0 and standard deviation as 1. So we are creating an object std_scl to use standardScaler.
    std_slc = StandardScaler()

    # We are also using Principal Component Analysis(PCA) which will reduce the dimension of features by creating new features which have most of the varience of the original data.
    pca = decomposition.PCA()

    # Here, we are using Decision Tree Classifier as a Machine Learning model to use GridSearchCV. So we have created an object dec_tree.
    dec_tree = tree.DecisionTreeClassifier()
    # Pipeline will helps us by passing modules one by one through GridSearchCV for which we want to get the best parameters. So we are making an object pipe to create a pipeline for all the three objects std_scl, pca and dec_tree.
    pipe = Pipeline(steps=[('std_slc', std_slc),
                           ('pca', pca),
                           ('dec_tree', dec_tree)])

    # Now we have to define the parameters that we want to optimise for these three objects.
    # StandardScaler doesnot requires any parameters to be optimised by GridSearchCV.
    # Principal Component Analysis requires a parameter 'n_components' to be optimised. 'n_components' signifies the number of components to keep after reducing the dimension.
    n_components = list(range(1, X.shape[1] + 1, 1))

    # DecisionTreeClassifier requires two parameters 'criterion' and 'max_depth' to be optimised by GridSearchCV. So we have set these two parameters as a list of values form which GridSearchCV will select the best value of parameter.
    criterion = ['gini', 'entropy']
    max_depth = [2, 4, 6, 8, 10, 12]

    # Now we are creating a dictionary to set all the parameters options for different objects.
    parameters = dict(pca__n_components=n_components,
                      dec_tree__criterion=criterion,
                      dec_tree__max_depth=max_depth)

    # Before using GridSearchCV, lets have a look on the important parameters.

    # estimator: In this we have to pass the models or functions on which we want to use GridSearchCV
    # param_grid: Dictionary or list of parameters of models or function in which GridSearchCV have to select the best.
    # Scoring: It is used as a evaluating metric for the model performance to decide the best hyperparameters, if not especified then it uses estimator score.

    # Making an object clf_GS for GridSearchCV and fitting the dataset i.e X and y
    clf_GS = GridSearchCV(pipe, parameters)
    clf_GS.fit(X, y)

    # Now we are using print statements to print the results. It will give the values of hyperparameters as a result.
    print('Best Criterion:', clf_GS.best_estimator_.get_params()['dec_tree__criterion'])
    print('Best max_depth:', clf_GS.best_estimator_.get_params()['dec_tree__max_depth'])
    print('Best Number Of Components:', clf_GS.best_estimator_.get_params()['pca__n_components'])
    print();
    print(clf_GS.best_estimator_.get_params()['dec_tree'])




elif max_value == 2:
    param_grid = {'C': [0.1, 1, 10, 100, 1000],
                  'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
                  'kernel': ['rbf']}

    grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=3)

    # fitting the model for grid search
    grid.fit(X_train, y_train)
    # print best parameter after tuning
    print(grid.best_params_)

    # print how our model looks after hyper-parameter tuning
    print(grid.best_estimator_)
    grid_predictions = grid.predict(X_test)

    # print classification report
    print(classification_report(y_test, grid_predictions))




elif max_value == 3:
    k_range = list(range(1, 31))
    param_grid = dict(n_neighbors=k_range)

    # defining parameter range
    grid = GridSearchCV(knn, param_grid, cv=10, scoring='accuracy', return_train_score=False, verbose=1)

    # fitting the model for grid search
    grid_search = grid.fit(x_train, y_train)



elif max_value == 4:
    pipe = make_pipeline((RandomForestClassifier()))

    grid_param = [
        {"randomforestclassifier": [RandomForestClassifier()],
         "randomforestclassifier__n_estimators": [10, 100, 1000],
         "randomforestclassifier__max_depth": [5, 8, 15, 25, 30, None],
         "randomforestclassifier__min_samples_leaf": [1, 2, 5, 10, 15, 100],
         "randomforestclassifier__max_leaf_nodes": [2, 5, 10]}]

    gridsearch = GridSearchCV(pipe, grid_param, cv=5, verbose=3, n_jobs=-1)
    best_model = gridsearch.fit(X_train, y_train)
    best_model.score(X_test, y_test)
    best_model.best_params_

# else:
# print("You messed up")
'''Best Model Score'''

best_model.score(X_test, y_test)

'''Best model parameters'''

best_model.best_params_

'''Confusion matrix with best params'''

# print classification report
print(classification_report(y_test, grid_predictions))

'''Additional Resources'''

'''Working out an error rate'''

# to work out the error rate, used to choose the correct amount of neighbours
error_rate = []

for i in range(1, 40)
    knn = KneighboursClassifier(n_neighbours=i)
    knn.fit(X_train, t_train)
    pred_i = knn.predict(X_test)
    error_rate.append(np.mean(pred_i != y_test))

plt.plot(range(1, 40), error_rate, colour='blue', linestyle='dashed', marker='o', markerfacecolor='red', markersize=10)
plt.title('Error rate vs k value')
plt.xlabel('k')
plt.ylabel('Error Rate')

'''PCA'''

######Why does this not work!!!! PCA Princible component anaylsis is used for removing non components from your data
scaler = StandardScaler()
scaler.fit(df)
scaled_data = scaler.transform(df)
pca = PCA(n_components=2)
pca.fit(scaled_data)
X_pca = pca.transform(scaled_data)

principalDf = pd.DataFrame(data=principalComponents
                           , columns=['principal component 1', 'principal component 2'])

X_pca.shape

scaled_data.shape

pca = PCA(n_components=2)
principalComponents = pca.fit_tranform(x)
principalDF = pd.Dataframe(data=principalComponents,
                           columns=['principal Component 1',
                                    'principal Component 2'])

'''KMeans'''

# This is for the artifical data make blobs

data = make_blobs(n_samples=200, n_features=2, centers=4, cluster_std=1.8, random_state=101)

plt.scatter(data[0][:, 0], data[0][:, 1], c=data[1],
            cmap="rainbow")  # Your plotting both rows of data against each other
data[0].shape

# Kmeans is used for finding clusters, not to acutally do anything withat material
# This is for the artifical data make blobs
Kmeans = KMeans(n_clusters=4)
Kmeans.fit(data[0])
Kmeans.cluster_centers_  # tuple of cluster centers
# THis is for finding labels not predicting labels
Kmeans.labels_  # an array of what it thinks our the cluster, predicted lbels
data[1]  # the correct labels

# This is for the artifical data make blobs
# this is for comapring the test data with the actual train data
fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(10, 6))
ax1.set_title('Kmeans')
ax1.scatter(data[0][:, 0], data[0][:, 1], c=Kmeans.labels_, cmap='rainbow')

ax2.set_title('Original')
ax2.scatter(data[0][:, 0], data[0][:, 1], c=data[1], cmap='rainbow')



