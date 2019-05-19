 #!/usr/bin/env python3
# -*- coding: utf-8 -*-

###################################################
# File: logistic_regresion_with_regularyzation.py #
#                                                 #
# Author: Marta Urbaniak                          #
#                                                 #
# Date: 17.05.2019                                #
#                                                 #
# Description:                                    #
#                                                 #    
#    Making predictions with                      #
#      logistic regresion                         #
#           model                                 #
#                                                 #
#    hipotesis: sigmoid function                  #
###################################################

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
    
    
data = loaddata("data/data2.txt",",")

y = np.c_[data[:,2]]

X = data[:,0:2]]

plotData(data, 'Test 1', 'Test 2', 'y=1', 'y=0')

#features mapping

poly = PolynomialFeatures(6)

XX = poly.fit_transform(data[:,0:2])

def sigmoid(z):
    
    s = 0
    
    s = 1/(1+np.exp(-z))
    
    return s
  
  
def cosFunctionReg(theta, reg, *args):
  
  m = y.size
  
  J = 0
  
  h = sigmoid(np.dot(XX, theta))
  
  J = 1/m * (-np.transpose(y),np.log(h))-np.dot(np.transpose(1-y),np.log(1-h))) + reg/2*m * np.sum(theta[1:]**2)
  
  if np.isnan(J[0]):
        return(np.inf)
   
  
  return(J[0])
  
def gradienReg(theta, teg, *args):
  
  m = y.size
  
  theta = theta.reshape(-1,1)
  
  h = sigmoid(np.dot(XX, theta))
  
  grad = 1/m * np.dot(np.transpose.(XX),h-y)
  
  theta[0] = 0
  
  reg_f = reg/m*theta
  
  grad = grad + reg_f
  
  return(grad.revel())
  
def predict(theta, X):
    
    p = 0
    
    p=sigmoid(np.dot(X,theta))
    
    p=p+0.5

    p=p.astype(int)
    
    
    return(p)

#checking cost and gradient value  
  
reg = 10

initial_theta = np.zeros((XX.shape[1], 1))

cost = costFunctionReg(initial_theta,reg, XX, y)

print(cost)

grad = gradientReg(initial_theta, reg, XX, y)

fig, axes = plt.subplots(1,3, sharey = True, figsize=(17,5))

#calculating best theta parametrs ( with minimize function)
#plotting decision boundry 

for i,C in enumerate([0,1,100]):
    
    res2 = minimize(costFunctionReg, initial_theta, args=(C, XX, y), method=None, jac=gradientReg, options={'maxiter':3000})
    
    #checking the precision
    
    dokladnosc = 100*sum(predict(res2.x, XX) == y.ravel())/y.size
    
    plotData(data2, 'Microchip Test 1', 'Microchip Test 2', 'y = 1', 'y = 0', axes.flatten()[i])
    
    x1_min, x1_max = X[:,0].min(), X[:,0].max(),
    x2_min, x2_max = X[:,1].min(), X[:,1].max(),
    
    xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max), np.linspace(x2_min, x2_max))
    
    h = sigmoid(poly.fit_transform(np.c_[xx1.ravel(), xx2.ravel()]).dot(res2.x))
    h = h.reshape(xx1.shape)
    
    axes.flatten()[i].contour(xx1, xx2, h, [0.5], linewidths=1, colors='g');       
    axes.flatten()[i].set_title('Dokładność trenowania {}% z Lambda = {}'.format(np.round(dokladnosc, decimals=2), C))
    