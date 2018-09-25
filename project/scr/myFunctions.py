import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import roc_auc_score
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

def mode(a):
    dict = {}
    entries = np.unique(a)
    for i in entries:
        dict[i] = len(a[a == i])
    return dict

def RF_gridSearch(Xtrain, ytrain, index_train, index_test):
    n_estimators = [100, 200, 300, 400]
    max_depth = [5, 10, 15, 20]
    min_samples_split = [2, 3, 4]
    res = np.zeros((len(n_estimators),len(max_depth),len(min_samples_split)))

    for index in range(len(index_train)):
        X_train = Xtrain[index_train[index], :]
        y_train = ytrain[index_train[index]]
        X_test = Xtrain[index_test[index], :]
        y_test = ytrain[index_test[index]]

        for i in range(len(n_estimators)):
            for j in range(len(max_depth)):
                for k in range(len(min_samples_split)):
                    forest = RandomForestRegressor(n_estimators=n_estimators[i],
                                                   random_state=0,
                                                   oob_score=True,
                                                   max_depth=max_depth[j],
                                                   min_samples_split=min_samples_split[k],
                                                   n_jobs=-1)
                    forest.fit(X_train, y_train)
                    output = forest.predict(X_test)
                    res[i][j][k] += roc_auc_score(y_test, output)
    return res / 5



def SVM_gridSearch(Xtrain, ytrain, index_train, index_test):
    C = [1, 5, 10, 100]
    gamma = [0.001, 0.002, 0.005, 0.1]
    res = np.zeros((len(C),len(gamma)))

    for index in range(len(index_train)):
        X_train = Xtrain[index_train[index], :]
        y_train = ytrain[index_train[index]]
        X_test = Xtrain[index_test[index], :]
        y_test = ytrain[index_test[index]]

        for i in range(len(C)):
            for j in range(len(gamma)):
                    clf_svm = svm.SVR(C=C[i],
                                      gamma=gamma[j])
                    clf_svm.fit(X_train, y_train)
                    output = clf_svm.predict(X_test)
                    res[i][j] += roc_auc_score(y_test, output)
    return res / 5



def KNN_gridSearch(Xtrain, ytrain, index_train, index_test):
    n_neighbors = [3, 5, 10, 15, 20, 25, 30, 40, 50]
    res = np.zeros((len(n_neighbors)))

    for index in range(len(index_train)):
        X_train = Xtrain[index_train[index], :]
        y_train = ytrain[index_train[index]]
        X_test = Xtrain[index_test[index], :]
        y_test = ytrain[index_test[index]]

        for i in range(len(n_neighbors)):
                clf_knn = KNeighborsClassifier(n_neighbors=n_neighbors[i])
                clf_knn.fit(X_train, y_train)
                output = clf_knn.predict_proba(X_test)[:, 1]
                res[i] += roc_auc_score(y_test, output)
    return res / 5


def getAccuracy(a, y_test):
    a[a >= 0.5] = 1
    a[a < 0.5] = 0
    return accuracy_score(y_test, a)


def plotROC(y_test, y_score):
    # Compute ROC curve and ROC area for each class

    fpr, tpr, threshold = roc_curve(y_test, y_score, pos_label=1)  ###计算真正率和假正率
    roc_auc = auc(fpr, tpr)  ###计算auc的值

    plt.figure()
    lw = 2
    plt.figure(figsize=(10, 10))
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)  ###假正率为横坐标，真正率为纵坐标做曲线
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve')
    plt.legend(loc="lower right")
    plt.show()
