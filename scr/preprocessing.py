import pandas as pd
import numpy as np
import myFunctions as mf
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import roc_auc_score
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
import xgboost as xgb


train = pd.read_csv("bank-additional.csv")

# -------------------pre-processing--------------------
# train = train[((train.default!="unknown") | (train.housing!="unknown") | (train.loan!="unknown")) & (train.default!="yes")]
train["y"] = train.y.apply(lambda x: 1 if x == "yes" else 0)

train["contact"] = train.contact.apply(lambda x: 1 if x == "cellular" else 0)

# counts = mf.mode(train["marital"])
train["marital"].replace("unknown", "married", inplace=True)
marital_dummies = pd.get_dummies(train.marital)
train = pd.concat([train,marital_dummies], axis=1)
train.drop(['marital', 'poutcome'], axis=1, inplace=True)

weekday_dummies = pd.get_dummies(train.day_of_week)
train = pd.concat([train, weekday_dummies], axis=1)
train.drop(['day_of_week'], axis=1, inplace=True)

# counts = mf.mode(train["job"])
train["job"].replace("unknown", "admin.", inplace=True)
job_dummies = pd.get_dummies(train.job)
train = pd.concat([train, job_dummies], axis=1)
train.drop(['job'], axis=1, inplace=True)

month_dummies = pd.get_dummies(train.month)
train = pd.concat([train, month_dummies], axis=1)
train.drop(['month'], axis=1, inplace=True)


def education_to_num(a):
    edu = ['basic.4y', 'basic.6y', 'basic.9y', 'high.school', 'illiterate', 'professional.course', 'university.degree']
    list = [0, 0.15, 0.3, 0.45, 0.6, 0.75, 1]
    for i in range(7):
        if (a == edu[i]):
            return list[i]

# 'education'
# counts = mf.mode(train["education"])
train["education"].replace("unknown", "university.degree", inplace=True)
train["education"] = train.education.apply(education_to_num)

# 'default'
# counts = mf.mode(train["default"])
default_dummies = pd.get_dummies(train.default)
train["default"] = train.default.apply(lambda x: 1 if x == "unknown" else 0)

# 'housing'
# counts = mf.mode(train["housing"])
train["housing"].replace("unknown", "yes", inplace=True)
train["housing"] = train.housing.apply(lambda x: 1 if x == "yes" else 0)

# 'loan'
# counts = mf.mode(train["loan"])
train["loan"].replace("unknown", "no", inplace=True)
train["loan"] = train.loan.apply(lambda x: 1 if x == "yes" else 0)

# divided label
label = train["y"]
train.drop(["y"], axis=1, inplace=True)

# train.to_csv('training_data/train.csv', index=None)


# split train & test data   X_test & y_test are used for test case
col = train.columns
X_train, X_test, y_train, y_test = train_test_split(train, label, test_size=0.2, random_state=42)
X_train = np.array(X_train)
X_test = np.array(X_test)
y_train = np.array(y_train)
y_test = np.array(y_test)

# normalize
scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)



# -------------------feature-selection--------------------
'''select = SelectKBest(chi2, k=35)
select.fit(X_train, y_train)
print(select.pvalues_ * 10000)'''

forest = RandomForestClassifier(n_estimators=100, random_state=0, n_jobs=-1)
forest.fit(X_train, y_train)
importance = forest.feature_importances_
output = forest.predict(X_test)
'''
name_list = train.columns
plt.barh(range(len(importance)), importance,color='rgb',tick_label=name_list)
plt.show()
'''

# reduce features
delete_index = [i for i in range(len(importance)) if importance[i] < 0.004]
X_train = np.delete(X_train, delete_index, axis=1)
X_test = np.delete(X_test, delete_index, axis=1)

#-------------------parameter adjustment-------------------------
# K-fold
skf = StratifiedKFold(n_splits=5, shuffle=False)

index_train = []
index_test = []

for train_index, test_index in skf.split(X_train, y_train):
    index_train.append(train_index)
    index_test.append(test_index)


#-------------------random forest-------------------------

# gridSearchResult = mf.RF_gridSearch(X_train, y_train, index_train, index_test)

rf_clf = RandomForestRegressor(n_estimators=300,
                                random_state=0,
                                oob_score=True,
                                max_depth=5,
                                n_jobs=-1,
                                min_samples_split=2)
rf_clf.fit(X_train, y_train)
rf_pred = rf_clf.predict(X_test)
fpr, tpr, thresholds = metrics.roc_curve(y_test, rf_pred, pos_label=1)
# mf.plotROC(y_test, rf_pred)
print("Random Forest AUC: ",metrics.auc(fpr, tpr))
print("Random Forest Accuracy: ", mf.getAccuracy(rf_pred, y_test))

#--------------------SVM-------------------------

# gridSearchResult = mf.SVM_gridSearch(X_train, y_train, index_train, index_test)

svm_clf = svm.SVR(C=10,
                  gamma=0.002)
svm_clf.fit(X_train, y_train)
svm_pred = svm_clf.predict(X_test)
fpr2, tpr2, thresholds2 = metrics.roc_curve(y_test, svm_pred, pos_label=1)
# mf.plotROC(y_test, svm_pred)
print("SVM AUC: ",metrics.auc(fpr2, tpr2))
print("SVM Accuracy: ", mf.getAccuracy(svm_pred, y_test))


#--------------------Naive Bayes-------------------------

gnb_clf = GaussianNB()
gnb_clf.fit(X_train, y_train)
gnb_pred = gnb_clf.predict_proba(X_test)[:, 1]
fpr3, tpr3, thresholds3 = metrics.roc_curve(y_test, gnb_pred, pos_label=1)
# mf.plotROC(y_test, gnb_pred)
print("Naive Bayes AUC: ",metrics.auc(fpr3, tpr3))
print("Naive Bayes Accuracy: ", mf.getAccuracy(gnb_pred, y_test))

#--------------------KNN-------------------------

# gridSearchResult = mf.KNN_gridSearch(X_train, y_train, index_train, index_test)

knn_clf = KNeighborsClassifier(n_neighbors=40)
knn_clf.fit(X_train, y_train)
knn_pred = knn_clf.predict_proba(X_test)[:, 1]
fpr4, tpr4, thresholds4 = metrics.roc_curve(y_test, knn_pred, pos_label=1)
# mf.plotROC(y_test, knn_pred)
print("KNN AUC: ", metrics.auc(fpr4, tpr4))
print("KNN Accuracy: ", mf.getAccuracy(knn_pred, y_test))



#---------------------xgboost-----------------------
data = xgb.DMatrix(X_train,label=y_train)
params={'booster':'gbtree',        # booster:{gbtree, gblinear}
	    'objective': 'binary:logistic',
	    'eval_metric':'auc',
	    'gamma':0.1,                # minimum loss reduction required to make a split
	    'max_depth':5,              # maximum tree depth
	    'lambda':3,                # L2 regular term in the object function
	    'subsample':0.7,
	    'colsample_bytree':0.7,     # rate of random choose columns
	    'colsample_bylevel':0.7,
	    'eta': 0.01,                # shrink parameter
	    'tree_method':'exact',      # 使用精确节点分裂的方法
	    'seed':0
	    }
watchlist = [(data,'train')]
model = xgb.train(params,dtrain=data,num_boost_round=100,evals=watchlist)
xg_test = xgb.DMatrix(X_test)
xgboost_pred = model.predict(xg_test)
fpr5, tpr5, thresholds5 = metrics.roc_curve(y_test, xgboost_pred, pos_label=1)
print("Xgboost AUC: ", metrics.auc(fpr5, tpr5))
print("Xgboost Accuracy: ", mf.getAccuracy(xgboost_pred, y_test))
