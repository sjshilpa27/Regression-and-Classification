import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.linear_model import SGDClassifier
from sklearn import metrics

data=pd.DataFrame(np.loadtxt("classification.txt",delimiter=","))

feature_names = [0,1,2]
X = data[feature_names]
y = data[3]

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

clf = SGDClassifier()
# fit (train) the classifier
clf.fit(X_train, y_train)

# print learned coeficients
print("Weights:- " + str(clf.coef_))
print("Intercept:- " + str(clf.intercept_))
print("\n")


y_train_pred = clf.predict(X_train)

print("Accuracy on test dataset:- " + str(metrics.accuracy_score(y_train, y_train_pred)))

#Measure accuracy on the testing set
y_pred = clf.predict(X_test)
print("Accuracy on test dataset:- " + str(metrics.accuracy_score(y_test, y_pred)))

print (metrics.classification_report(y_test, y_pred))

print("Confusion Metrics")
print (metrics.confusion_matrix(y_test, y_pred))

ones = np.ones((X.shape[0],1))
plot_x = np.hstack((ones,X))

X_train = plot_x[:,1:]
Y_train = y
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
w = clf.coef_[0]
coefs = w
intercept = clf.intercept_[0]
temp = np.matrix(np.linspace(0,1,2))
xs = np.tile(temp,(temp.shape[0],1))
ys = np.tile(temp,(temp.shape[0],1)).T
zs = (xs*coefs[0]+ys*coefs[1]+intercept) * -1/coefs[2]
print("Equation: {:.2f} + {:.2f}x1 + {:.2f}x2 + {:.2f}x3 = 0".format(intercept, coefs[0],
                                                          coefs[1],coefs[2]))

ax.plot_surface(xs,ys,zs, alpha=0.5)
plt.show()