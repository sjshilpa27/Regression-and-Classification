#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


data=pd.DataFrame(np.loadtxt("classification.txt",delimiter=","))

feature_names = [0,1,2]
X = data[feature_names]
y = data[4]

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

#Calling Logistic Regression Model
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
print('Accuracy of Logistic regression classifier on training set: {:.2f}'
     .format(logreg.score(X_train, y_train)))
print('Accuracy of Logistic regression classifier on test set: {:.2f}'
     .format(logreg.score(X_test, y_test)))

y_pred = logreg.predict(X_test)
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(X_test, y_test)))

confusion_matrix = confusion_matrix(y_test, y_pred)
print(confusion_matrix)

print(classification_report(y_test, y_pred))
print("Weights")
print(logreg.coef_)

logit_roc_auc = roc_auc_score(y_test, logreg.predict(X_test))
fpr, tpr, thresholds = roc_curve(y_test, logreg.predict_proba(X_test)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig('Log_ROC')

ones = np.ones((X.shape[0],1))
plot_x = np.hstack((ones,X))

X_train = plot_x[:,1:]
Y_train = data[4]
c0 = c1 = 0 # Counter of label -1 and label 1 instances
for i in range(0, X_train.shape[0]):
    if Y_train[i] == -1:
        c0 = c0 + 1
    else:
        c1 = c1 + 1
x0 = np.ones((c0,3)) # matrix label -1 instances
x1 = np.ones((c1,3)) # matrix label 1 instances
k0 = k1 = 0
for i in range(X_train.shape[0]):
    if Y_train[i] == -1:
        x0[k0] = X_train[i]
        k0 = k0 + 1
    else:
        x1[k1] = X_train[i]
        k1 = k1 + 1

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(x0[:,0], x0[:,1], x0[:,2], marker='.', color = 'red')
ax.scatter(x1[:,0], x1[:,1], x1[:,2], marker='.', color = 'green')

ax.set_xlabel("X1")
ax.set_ylabel("X2")
ax.set_zlabel("X3")
w = logreg.coef_[0]
print(w)
coefs = w
print(logreg.intercept_)
intercept = logreg.intercept_[0]
temp = np.matrix(np.linspace(0,1,2))
xs = np.tile(temp,(temp.shape[0],1))
ys = np.tile(temp,(temp.shape[0],1)).T
zs = (xs*coefs[0]+ys*coefs[1]+intercept) * -1/coefs[2]
print("Equation: {:.2f} + {:.2f}x1 + {:.2f}x2 + {:.2f}x3 = 0".format(intercept, coefs[0],
                                                          coefs[1],coefs[2]))

ax.plot_surface(xs,ys,zs, alpha=0.5)
plt.show()
