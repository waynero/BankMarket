import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.metrics import classification_report

#
# train_line = np.loadtxt('wine_train.csv', delimiter=',', dtype='float')
# train_feature = train_line[:, 0:2]
# train_label = train_line[:, 13]
#
# test_line = np.loadtxt('wine_test.csv', delimiter=',', dtype='float')
# test_feature = test_line[:, 0:  2]
# test_label = test_line[:, 13]

train_feature = []
train_label = []
test_feature = []
test_label = []

# normalization

scaler = StandardScaler()
scaler.fit(train_feature)
train_feature_norm = scaler.transform(train_feature)
test_feature_norm = scaler.transform(test_feature)

# print('The unnormalized mean', np.mean(train_feature, axis = 0))
# print('The unnormalized std:', np.std(train_feature, axis=0))


# ppn = Perceptron(n_iter = 50, eta0 = 1, random_state = 0)

ppn = Perceptron()
ppn.fit(train_feature_norm, train_label)
print('Perceptron train error rate is', ppn.score(train_feature_norm, train_label))
print('Perceptron test error rate is: ', ppn.score(test_feature_norm, test_label))


pre_label = ppn.predict(test_feature)
ans = classification_report(test_label, pre_label)
print(ans)