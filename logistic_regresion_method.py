#!/usr/bin/env python3
# -*- coding: utf-8 -*-

######################################
# File: logistic_regresion_method.py #
#                                    #
# Author: Marta Urbaniak             #
#                                    #
# Date: 13.05.2019                   #
#                                    #
# Description:                       #
#                                    #    
#    Making predictions with         #
#      logistic regresion            #
#           model                    #
#                                    #
#    hipotesis: sigmoid function     #
######################################

import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from scipy.optimize import minimize

from sklearn.preprocessing import PolynomialFeatures



def loaddata(file, delimeter):
    
    data = np.loadtxt(file, delimiter=delimeter)
    
    print("Wymiar danych treningowych: ", data.shape)
    print(data[1:6,:])
    
    return(data)
  
def plotData(data, label_x, label_y, label_pos, label_neg, axes=None):
    
    neg = data[:,2] == 0
    pos = data[:,2] == 1
    
    
    if axes == None:
        axes = plt.gca()
    
    axes.scatter(data[pos][:,0], data[pos][:,1], marker='+', c='g', s=60, linewidth=2, label=label_pos)
    axes.scatter(data[neg][:,0], data[neg][:,1], c='r', s=60, label=label_neg)
    axes.set_xlabel(label_x)
    axes.set_ylabel(label_y)
    axes.legend(frameon= True, fancybox = True);
    
    plt.savefig("data_visualisation/data.jpg")
    
    
data = loaddata("data/data.txt",",")

y = np.c_[data[:,2]]
m = y.size
X = np.c_[np.ones((m,1)), data[:,0:2]]

#ploting the data

plotData(data, 'Wynik 1-go egzaminu', 'Wynik 2-go Egzaminu', 'Przyjęty', 'Nie przyjęty')

def sigmoid(z):
    
    s = 0
    
    s = 1/(1+np.exp(-z))
    
    return s
  
  
def costFunction(theta, x, y):
    
    h = 0
    
    h = sigmoid(np.dot(X, theta))
    
    J=1/m*(np.dot(-np.transpose(y),np.log(h))-np.dot(np.transpose(1-y),np.log(1-h)))
    
    if np.isnan(J[0]):
        return(np.inf)
    
    return J
    
    
def gradient(theta, X, y):
    
    m = y.size
    
    h=0
    
    theta=theta.reshape(-1,1)
    
    h=sigmoid(np.dot(X,theta))
    
    grad =1/m*(np.dot(np.transpose(X),h-y) )

    return(grad.ravel())

initial_theta = np.zeros((X.shape[1], 1))
cost = costFunction(initial_theta, X, y)
grad = gradient(initial_theta, X, y)

print('Cost: \n', cost, type(cost))
print('Grad: \n', grad, type(grad), grad.shape)

#finding the best theta parametrs

res = minimize(costFunction, initial_theta, args=(X,y), method=None, jac=gradient, options={'maxiter':400})
res

theta = res.x.reshape(3,1)

#checking the resuts

def predict(theta, X):
  
  p = 0
  
  p = sigmoid(np.dot(X,theta)
	      
  p = p + 0.5
  
  p = p.astype(int)
  
  return(p)

p = predict(theta, X)

dokladnosc = 0

dokladnosc = np.mean(p==y)

print("Dokladnosc {}%". format(dokladnosc*100))

pp = 0

W=[1,45,85]

pp=sigmoid(np.dot(W,theta))

print(("Student ma %2.2f%% szans na to, że dostanie się na studia")%(pp*100))

#ploting the decision boundary

plt.scatter(45,85, s = 60, c ="r", marker = "v", label = "(45,85)" )
plotData(data, 'Wynik 1-go egzaminu', 'Wynik 2-go egzaminu', 'Przyjęty', 'Nie przyjęty')

x1_min, x1_max = X[:,1].min(), X[:,1].max()
args = np.array([x1_min, x1_max])


y = 0

y_1=-args[0]*(theta[1]/theta[2])-theta[0]/theta[2]
y_2=-args[1]*(theta[1]/theta[2])-theta[0]/theta[2]

y=[y_1,y_2]

plt.plot(args, y)
plt.savefig("data_visualisation/dec_boundary.png")
	      
	      