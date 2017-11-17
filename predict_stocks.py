# -*- coding: utf-8 -*-
"""
Created on Fri Nov 17 10:41:34 2017

@author: Shubham
"""

import csv
import numpy as np
from sklearn.svm import SVR
import matplotlib.pyplot as plt

dates=[]
prices=[]

def get_data():
    with open('aapl.csv', 'r') as file:
        reader= csv.reader(file)
        next(reader)
        for row in reader:
    #        print(row[0])
            dates.append(int(row[0].split('-')[2]))
            prices.append(float(row[1]))
    return

def predict(dates, prices):
    dates= np.reshape(dates, (len(dates), 1))
    svr_lin  =  SVR(kernel= 'linear', C=1e3)
    svr_poly =  SVR(kernel= 'poly', C=1e3, degree= 2)
    svr_rbf  =  SVR(kernel='rbf', C= 1e3, gamma=0.1)
    print('Model created')
    svr_lin.fit(dates, prices)
    svr_poly.fit(dates, prices)
    svr_rbf.fit(dates, prices)
#    svr_poly.predict(dates)

    plt.scatter(dates, prices, color='black', label='Data')
    plt.plot(dates, svr_lin.predict(dates), color='red', label= 'Linear')
    plt.plot(dates, svr_poly.predict(dates), color='blue', label= 'Poly')

    plt.plot(dates, svr_rbf.predict(dates), color='green', label= 'RBF')
    plt.ylabel('Price')
    plt.xlabel('Date')    
    plt.title('SVR')
    plt.legend()
    plt.show()

get_data()
predict(dates, prices)    
 