import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

df = pd.read_csv('china_gdp.csv')

x_data = df['Year'].values
y_data = df['Value'].values

def sigmoid(x, beta1, beta2):
    y = 1/(1 + np.exp(-beta1*(x-beta2)))
    return y

x = np.arange(1960,2015,1)
x = x/max(x)
beta1 = 0
beta2 = 0

xdata = x_data/max(x_data)
ydata = y_data/max(y_data)

popt, pcov = curve_fit(sigmoid,xdata,ydata)

plt.figure(figsize=(8,5))
y = sigmoid(x, *popt)
plt.plot(xdata,ydata,'ro',label='data')
plt.plot(x,y, linewidth=3.0, label='fit')
plt.legend(loc='best')
plt.ylabel('GDP')
plt.xlabel('Year')
plt.show()