'''
    PA-4: Linear Classification, Linear Regression & Logistic Regression
    Authors:
    Amitabh Rajkumar Saini, amitabhr@usc.edu
    Shilpa Jain, shilpaj@usc.edu
    Sushumna Khandelwal, sushumna@usc.edu
    Dependencies:
    1. numpy : pip install numpy
    2. matplotlib : pip install matplotlib
    Output:
    Returns linear classifier, logistic regression and linear regression models with graphs and metrics.
'''
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class linearClassifier:
    '''
        Implements Perceptron Learning and Pocket Learning Algorithm
    '''
    def __init__(self, x, y, alpha, threshold=0.002,iterations=1000):
        '''
            Constructs a linear classifier object
            :param x: input dataset as numpy array
            :param y: actual outputs as numpy array
            :alpha: learning rate
            :threshold: threshold for misclassifiction
            :interations: number of epochs to run, default value being 1000
            :return: returns nothing
        '''
        ones = np.ones((x.shape[0],1))
        self.X = np.hstack((ones,x))
        self.Y = y
        self.alpha = alpha
        self.w = np.random.rand(1,self.X.shape[1])
        self.threshold = threshold
        self.iterations = iterations
    
    def activation(self,x):
        '''
            calculates activation of a data point value using sign function
            :param x: a data point
            returns: activation function value
        '''
        z = np.dot(self.w,x.T)
        if z[0] < 0:
            return -1
        else:
            return 1

    def train(self):
        '''
            builds a classifier based on perceptron learning algorithm
            returns: nothing
        '''
        violated_constraint = True
        counter = 0
        t = float('inf')
        print("Misclassification Threshold:",self.threshold)
        while violated_constraint and t>self.threshold:
            violated_constraint = False
            miscount = 0
            for i, point in enumerate(self.X):
                if self.activation(point) != self.Y[i]:
                    violated_constraint = True
                    self.w = self.w + (self.Y[i]*self.alpha)*point
                    miscount +=1
            t = miscount/self.X.shape[0]
            counter +=1
            print("Misclassification: %f" % (t), end="\r")
        print("Weights:",self.w)
        print("Misclassifications:",miscount)

    def pocket_train(self):
        '''
            builds a classifier based on pocket learning algorithm
            returns: nothing
        '''
        violated_constraint = True
        min_violation_count = float('inf')
        pocket_w = None
        j = 0
        print("Iterations:",self.iterations)
        while j < self.iterations and violated_constraint:
            violated_constraint = False
            violation_count = 0
            for i, point in enumerate(self.X):
                if self.activation(point) != self.Y[i]:
                    violated_constraint = True
                    violation_count+=1
                    self.w = self.w + (self.Y[i]*self.alpha)*point
            if violation_count < min_violation_count:
                min_violation_count = violation_count
                pocket_w = self.w
            print("Epoch: %d        Min Misclassification: %d" % (j,min_violation_count), end="\r")
            j += 1

        self.w = pocket_w
        print("\nWeights:",pocket_w)
        print("Misclassification:",min_violation_count)

    def predict(self,x_test_data):
        '''
            predicts the classification of the data points received in parameters
            :params x: input values for the model
            :returns: predictes classification as numpy array
        '''
        
        predicted_y = []
        for i in range(x_test_data.shape[0]):
            z = np.dot(self.w[:,1:],x_test_data[i].T)[0]+self.w[0]
            if z[0] < 0:
                predicted_y.append(-1)
            else:
                predicted_y.append(1)
        return np.asarray(predicted_y)



    def plot(self):
        '''
           plots the data points along with the corresponding classifiying plane
        '''
        X_train = self.X[:,1:]
        Y_train = self.Y
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
        w = self.w[0]
        coefs = w[1:]
        intercept = w[0]
        temp = np.matrix(np.linspace(0,1,2))
        xs = np.tile(temp,(temp.shape[0],1))
        ys = np.tile(temp,(temp.shape[0],1)).T
        zs = (xs*coefs[0]+ys*coefs[1]+intercept) * -1/coefs[2]
        print("Equation: {:.2f} + {:.2f}x1 + {:.2f}x2 + {:.2f}x3 = 0".format(intercept, coefs[0],
                                                                  coefs[1],coefs[2]))

        ax.plot_surface(xs,ys,zs, alpha=0.5)
        plt.show()


class linearRegression:
    '''
        Implements Linear Regression Algorithm
    '''
    def __init__(self, X, Y):
        '''
            Constructs a linear classifier object
            :param x: input dataset as numpy array
            :param y: actual outputs as numpy array
            :return: returns nothing
        '''
        ones = np.ones((X.shape[0],1))
        self.x = np.hstack((ones,X))
        self.y = Y
        self.w = None

    def train(self):
        '''
            builds a models based on linear regression algorithm
            returns: nothing
        '''
        self.w = np.linalg.inv(self.x.T.dot(self.x)).dot(self.x.T.dot(self.y))
        print("Weights:",self.w)
       
    def predict(self,X):
        '''
            predicts the output of the data points received in parameters
            :params x: input values for the model
            :returns: numpy array of predicted values
        '''
        return np.array(self.w[0] + self.w[1] * X[:,0] + self.w[2] * X[:,1])

    def plot(self):
        '''
            plots the data points along with the corresponding plane
        '''
        X_train = self.x[:,1:]
        y_train = self.y
        X_test = X_train[:100,:]
        y_test = y_train[:100]

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(X_train[:,0], X_train[:,1], y_train, marker='.', color='red')
        ax.set_xlabel("X1")
        ax.set_ylabel("X2")
        ax.set_zlabel("y")

        y_pred = self.predict(X_test)

        print("MAE: {}".format(np.abs(y_test-y_pred).mean()))
        print("RMSE: {}".format(np.sqrt(((y_test-y_pred)**2).mean())))

        coefs = self.w[1:]
        intercept = self.w[0]
        temp = np.linspace(0,1,2)
        xs = np.tile(temp,(temp.shape[0],1))
        ys = np.tile(temp,(temp.shape[0],1)).T
        zs = xs*coefs[0]+ys*coefs[1]+intercept
        print("Equation: y = {:.2f} + {:.2f}x1 + {:.2f}x2".format(intercept, coefs[0],
                                                                  coefs[1]))
        ax.plot_surface(xs,ys,zs, alpha=0.5)
        plt.show()
    

class logisticRegression:
    '''
        Implements Logistic Regression Algorithm
    '''
    def __init__(self, X, Y, alpha, iterations,threshold=0.001):
        '''
            Constructs a logistic regression object
            :param x: input dataset as numpy array
            :param y: actual outputs as numpy array
            :alpha: learning rate
            :threshold: threshold for misclassifiction
            :interations: number of epochs to run, default value being 1000
            :return: returns nothing
        '''
        ones = np.ones((X.shape[0],1))
        self.x = np.hstack((ones,X))
        self.y = Y
        self.y = self.y[:, np.newaxis]
        self.alpha = alpha
        self.iterations = iterations
        self.w = np.random.rand(1,self.x.shape[1])

    def sigmoid(self,s):
        '''
            calculates activation of a data point value using sigmoid function
            :param s: a data point
            returns: activation function value
        '''
        return (np.exp(s))/(1+np.exp(s))

    def loss_fn(self):
        '''
           calculates loss fn associated with the model
           returns: loss value
        '''
        n = self.y.shape[0]
        sum = 0
        for i in range(n):
            p = - self.y[i] * np.dot(self.w,self.x[i].T)
            sum += np.log(1+np.exp(p))
        sum=sum/n
        return sum

    def train(self):
        '''
            builds a models based on logistic regression algorithm
            returns: nothing
        '''
        loss = self.loss_fn()
        print("Loss Value:",loss)
        n = self.y.shape[0]
        i = 0
        print("Iterations:",self.iterations)
        while i < self.iterations:
            term = np.matmul(self.x,self.w.T)
            ywtx = np.multiply(term,self.y)
            exp = 1/(1+np.exp(ywtx))
            prod = np.multiply(exp,np.multiply(self.x,self.y))
            summ = np.sum(prod,axis=0)
            del_w = summ/n
            self.w = self.w + self.alpha*del_w
            i+=1
        
        loss = self.loss_fn()
        print("Loss Value:",loss)
        print("Weights:",self.w)

    def predict(self,x_test_data):
        '''
            predicts the output of the data points received in parameters
            :params x: input values for the model
            :returns: numpy array of predicted values
        '''

        predicted_y = []
        for i in range(x_test_data.shape[0]):
            z = np.dot(self.w[:,1:],x_test_data[i].T)[0]+self.w[0]
            if self.sigmoid(z[0]) >= 0.5:
                predicted_y.append(1)
            else:
                predicted_y.append(-1)
        return np.asarray(predicted_y)


    def plot(self):
        '''
            plots the data points along with the corresponding plane
        '''
        X_train = self.x[:,1:]
        Y_train = self.y
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
        w = self.w[0]
        coefs = w[1:]
        intercept = w[0]
        temp = np.matrix(np.linspace(0,1,2))
        xs = np.tile(temp,(temp.shape[0],1))
        ys = np.tile(temp,(temp.shape[0],1)).T
        zs = (xs*coefs[0]+ys*coefs[1]+intercept) * -1/coefs[2]
        print("Equation: {:.2f} + {:.2f}x1 + {:.2f}x2 + {:.2f}x3 = 0".format(intercept, coefs[0],
                                                                  coefs[1],coefs[2]))

        ax.plot_surface(xs,ys,zs, alpha=0.5)
        plt.show()

def metrics(actual, predicted):
    '''
        Calculates metrics
        :param actual: actual values in list
        :param predicted: model predicted values in list
        :return : returns nothing
    '''
    confusion_matrix = [[0, 0], [0, 0]]
    #(predicted - actual)
    for i in range(len(actual)):
        
        actual[i] = 1 if actual[i] == 1 else 0
        predicted[i] = 1 if predicted[i] == 1 else 0
        
        if actual[i] and predicted[i]:
            confusion_matrix[0][0] += 1
        elif actual[i] and not predicted[i]:
            confusion_matrix[0][1] += 1
        elif not actual[i] and predicted[i]:
            confusion_matrix[1][0] += 1
        else:
            confusion_matrix[1][1] += 1

    accuracy = (confusion_matrix[1][1] + confusion_matrix[0][0]) / (
                                                                sum(confusion_matrix[0]) + sum(confusion_matrix[1]))
    recall = (confusion_matrix[0][0]) / sum(confusion_matrix[0])
    precision = (confusion_matrix[0][0]) / (confusion_matrix[0][0] + confusion_matrix[1][0])
    Fmeasure = (2 * recall * precision) / (recall + precision)
    print("Accuracy: ", accuracy)
    print("Recall: ", recall)
    print("Precision: ", precision)
    print("Fmeasure: ", Fmeasure)
    print("=======================")
    Matrix = []
    Matrix.append(["", "P1", "N0"])
    Matrix.append((["P1"] + confusion_matrix[0]))
    Matrix.append((["N0"] + confusion_matrix[1]))
    for i in Matrix:
        print(i)
    print("TP: ", confusion_matrix[0][0])
    print("FP: ", confusion_matrix[0][1])
    print("FN: ", confusion_matrix[1][0])
    print("TN: ", confusion_matrix[1][1])


def main():
    '''
        Runner Program
        :return: returns nothing
    '''
    data = np.loadtxt("/Users/shilpajain/Downloads/Assignment4/classification.txt",delimiter=',')
    print("------------------------------------------")
    print("Perceptron Learning Algorithm")
    np.random.shuffle(data)
    instances = data.shape[0]
    training_data, test_data = data[:int(0.8 * instances), :], data[int(0.8 * instances):, :]
    lc = linearClassifier(training_data[:,:3],training_data[:,3],0.05)
    print("Number of Training Data", training_data.shape[0])
    print("Number of Test Data", test_data.shape[0])
    lc.train()
    predicted_y = lc.predict(test_data[:,:3])
    metrics(test_data[:,3],predicted_y)
    lc.plot()
    print("------------------------------------------")
    print("Pocket Algorithm")
    training_data, test_data = data[:int(0.8 * instances), :], data[int(0.8 * instances):, :]
    plc = linearClassifier(training_data[:,:3],training_data[:,4],0.05,iterations=7000)
    print("Number of Training Data", training_data.shape[0])
    print("Number of Test Data", test_data.shape[0])
    plc.pocket_train()
    predicted_y = lc.predict(test_data[:, :3])
    metrics(test_data[:, 4], predicted_y)
    plc.plot()
    print("------------------------------------------")
    print("Logistic Regression")
    training_data, test_data = data[:int(0.8 * instances), :], data[int(0.8 * instances):, :]
    logr = logisticRegression(training_data[:,:3],training_data[:,4],0.05,7000)
    print("Number of Training Data", training_data.shape[0])
    print("Number of Test Data", test_data.shape[0])
    logr.train()
    predicted_y = logr.predict(test_data[:, :3])
    metrics(test_data[:, 4], predicted_y)
    logr.plot()
    print("------------------------------------------")
    print("Linear Regression")
    data_lr = np.loadtxt("/Users/shilpajain/Downloads/Assignment4/linear-regression.txt",delimiter=',')
    lr = linearRegression(data_lr[:,:2],data_lr[:,2])
    print("Number of Training Data", training_data.shape[0])
    print("Number of Test Data", test_data.shape[0])
    lr.train()
    lr.plot()
    print("------------------------------------------")

if __name__ == "__main__":
    main()
