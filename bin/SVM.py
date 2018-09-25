import numpy as np
from sklearn.utils import shuffle
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
# from plotSVMBoundaries import plotDecBoundaries
from sklearn.model_selection import StratifiedKFold

train_line = np.loadtxt('/Users/weiyiliu/Desktop/EE559/hw10/wine_csv/feature_train.csv', delimiter=',', dtype='float')
x = train_line[:, 0:2]
y = np.loadtxt('/Users/weiyiliu/Desktop/EE559/hw10/wine_csv/label_train.csv', delimiter=',', dtype='float')

Cs = np.logspace(-3, 3, 50)
Gas = np.logspace(-3, 3, 50)

all_acc = np.zeros((50, 50))
std_acc = np.zeros((50, 50))
outC = []
outG = []
for time in range(0, 20):
    max1 = 0
    max2 = 0
    for i in range(0, 50):
        for j in range(0, 50):
            skf = StratifiedKFold(n_splits=5, shuffle=True)
            part_acc = []
            for train_index, dev_index in skf.split(x, y):
                x_train, x_dev = x[train_index], x[dev_index]
                y_train, y_dev = y[train_index], y[dev_index]
                clf = SVC(C=Cs[i], gamma=Gas[j], kernel='rbf')
                clf.fit(x_train, y_train);
                y_pre = clf.predict(x_dev)
                acc = accuracy_score(y_dev, y_pre)
                part_acc.append(acc)
            mean_ = np.mean(part_acc)
            if (mean_ > max1):
                max1 = mean_
                tep_C = Cs[i]
                tep_G = Gas[j]
            all_acc[i][j] = mean_
            std_ = np.std(part_acc)
            std_acc[i][j] = std_
    outC.append(tep_C)
    outG.append(tep_G)
    if (max2 < max1):
        max2 = max1
        fin_C = tep_C
        fin_G = tep_G

skf = StratifiedKFold(n_splits=5, shuffle=True)
part_acc = []
for train_index, dev_index in skf.split(x, y):
    x_train, x_dev = x[train_index], x[dev_index]
    y_train, y_dev = y[train_index], y[dev_index]
    clf = SVC(C=fin_C, gamma=fin_G, kernel='rbf')
    clf.fit(x_train, y_train);
    y_pre = clf.predict(x_dev)
    acc = accuracy_score(y_dev, y_pre)
    part_acc.append(acc)
mean_ = np.mean(part_acc)
std_ = np.std(part_acc)
print('final mean accuracy:', mean_, 'final mean std:', std_)

# test_line = np.loadtxt('/Users/weiyiliu/Desktop/EE559/hw10/wine_csv/feature_test.csv', delimiter=',', dtype='float')
# test = test_line[:, 0:2]
# test_y = np.loadtxt('/Users/weiyiliu/Desktop/EE559/hw10/wine_csv/label_test.csv', delimiter=',', dtype='float')

print(fin_C, fin_G)
clf = SVC(C=fin_C, gamma=fin_G, kernel='rbf')
clf.fit(x, y)
# plotDecBoundaries(x,y,clf)
y__pre = clf.predict(test)
acc = accuracy_score(test_y, y__pre)
print(acc)
