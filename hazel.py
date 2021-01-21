#We are importing all necessary libraries to implement our model
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

class LogisticRegression(object):
    
    def __init__(Logreg, alpha=0.01, n_iteration=100):  #This function intializes the alpha value and iteration 
        Logreg.alpha = alpha                            #value in the object
        Logreg.n_iter = n_iteration
        
    def _sigmoid_function(Logreg, x): #This function is resonsible for calculating the sigmoid value with given parameter
        value = 1 / (1 + np.exp(-x))
        return value
    def _cost_function(Logreg,h,theta, y): # The fuctions calculates the cost value
        m = len(y)
        cost = (1 / m) * (np.sum(-y.T.dot(np.log(h)) - (1 - y).T.dot(np.log(1 - h))))
        return cost
    
    def _gradient_descent(Logreg,X,h,theta,y,m): # This function calculates the theta value by gradient descent
        gradient_value = np.dot(X.T, (h - y)) / m
        theta -= Logreg.alpha * gradient_value
        return theta
    
    def fit(Logreg, X, y): #This function primarily calculates the optimal theta value using which we predict the future data
        print("Fitting the given dataset..")
        Logreg.theta = []
        Logreg.cost = []
        X = np.insert(X, 0, 1, axis=1)
        m = len(y)
        for i in np.unique(y): 
            #print('Descending the gradient for label type ' + str(i) + 'vs Rest')
            y_onevsall = np.where(y == i, 1, 0)
            theta = np.zeros(X.shape[1])
            cost = []
            for _ in range(Logreg.n_iter):
                z = X.dot(theta)
                h = Logreg._sigmoid_function(z)
                theta = Logreg._gradient_descent(X,h,theta,y_onevsall,m)
                cost.append(Logreg._cost_function(h,theta,y_onevsall)) 
            Logreg.theta.append((theta, i))
            Logreg.cost.append((cost,i))
        return Logreg

    def predict(Logreg, X): # this function calls the max predict function to classify the individul feauter
        X = np.insert(X, 0, 1, axis=1)
        X_predicted = [max((Logreg._sigmoid_function(i.dot(theta)), c) for theta, c in Logreg.theta)[1] for i in X ]

        return X_predicted

    def score(Logreg,X, y): #This function compares the predictd label with the actual label to find the model performance
        score = sum(Logreg.predict(X) == y) / len(y)
        return score
    
    def _plot_cost(Logreg,costh): # This function plot the Cost function value
        for cost,c in costh   :
                plt.plot(range(len(cost)),cost,'r')
                plt.title("Convergence Graph of Cost Function of type-" + str(c) +" vs All")
                plt.xlabel("Number of Iterations")
                plt.ylabel("Cost")
                plt.show()
                
# We are reading and processing the data provided
filename = 'hazelnuts.txt'
data = pd.read_csv(filename,sep='\t', header=None,dtype=str)
#Transposing the data
data_T = data.T
data_T = data_T.dropna()
data_T.dtypes
data_T.columns = ['sample_id','length', 'width','thickness','surface_area',
                  'mass','compactness','hardness','shell_top_radius','water_content','carbohydrate_content','variety']

y_data = data_T['variety'].values  #segregating the label vlue from the feature value.
X = data_T.drop(['variety','sample_id'],axis=1).values

scaler = StandardScaler()
X= scaler.fit_transform(X)
print(X)
scores=[]
from sklearn.model_selection import train_test_split
for _ in range (10):
    X_train,X_test,y_train,y_test = train_test_split(X,y_data,test_size = 0.33)
    logi = LogisticRegression(n_iteration=30000).fit(X_train, y_train)
    predition1 = logi.predict(X_test)
    score1 = logi.score(X_test,y_test)
    print("the accuracy of the model is ",score1)
    scores.append(score1)
    
print(np.mean(scores))
logi._plot_cost(logi.cost)