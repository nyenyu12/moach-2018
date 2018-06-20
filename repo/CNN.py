#Imports
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
import os

#Creating labels for data
types_d=np.identity(3)

#Getting data from files, normalising and then labeling it. The data is for sub 0.1, sub 0.5 and sub 0.9
Sub01_x=np.genfromtxt(os.path.dirname(os.path.realpath(__file__))+'/Sub01_x.txt',skip_footer=9000).T
a_min=np.amin(Sub01_x,axis=1).reshape(1000,1)
a_max=np.amax(Sub01_x,axis=1).reshape(1000,1)
Sub01_x=(Sub01_x-a_min)/(a_max-a_min)

Sub01_y=np.genfromtxt(os.path.dirname(os.path.realpath(__file__))+'/Sub04_y.txt',skip_footer=9000).T
a_min=np.amin(Sub01_y,axis=1).reshape(1000,1)
a_max=np.amax(Sub01_y,axis=1).reshape(1000,1)
Sub01_y=(Sub01_y-a_min)/(a_max-a_min)

data_Sub01=np.column_stack((Sub01_x,Sub01_y))
data_Sub01=np.column_stack((data_Sub01,np.tile(types_d[0],(1000,1))))
Sub01_x=None
Sub01_y=None

Sub05_x=np.genfromtxt(os.path.dirname(os.path.realpath(__file__))+'/Sub05_x.txt',skip_footer=9000).T
a_min=np.amin(Sub05_x,axis=1).reshape(1000,1)
a_max=np.amax(Sub05_x,axis=1).reshape(1000,1)
Sub05_x=(Sub05_x-a_min)/(a_max-a_min)

Sub05_y=np.genfromtxt(os.path.dirname(os.path.realpath(__file__))+'/Sub05_y.txt',skip_footer=9000).T
a_min=np.amin(Sub05_y,axis=1).reshape(1000,1)
a_max=np.amax(Sub05_y,axis=1).reshape(1000,1)
Sub05_y=(Sub05_y-a_min)/(a_max-a_min)

data_Sub05=np.column_stack((Sub05_x,Sub05_y))
data_Sub05=np.column_stack((data_Sub05,np.tile(types_d[1],(1000,1))))
Sub05_x=None
Sub05_y=None

Sub09_x=np.genfromtxt(os.path.dirname(os.path.realpath(__file__))+'/Sub09_x.txt',skip_footer=9000).T
a_min=np.amin(Sub09_x,axis=1).reshape(1000,1)
a_max=np.amax(Sub09_x,axis=1).reshape(1000,1)
Sub09_x=(Sub09_x-a_min)/(a_max-a_min)

Sub09_y=np.genfromtxt(os.path.dirname(os.path.realpath(__file__))+'/Sub09_y.txt',skip_footer=9000).T
a_min=np.amin(Sub09_y,axis=1).reshape(1000,1)
a_max=np.amax(Sub09_y,axis=1).reshape(1000,1)
Sub09_y=(Sub09_y-a_min)/(a_max-a_min)

data_Sub09=np.column_stack((Sub09_x,Sub09_y))
data_Sub09=np.column_stack((data_Sub09,np.tile(types_d[2],(1000,1))))
Sub09_x=None
Sub09_y=None

#Combining data into singel array and shuffling it
data=np.row_stack((data_Sub01,data_Sub05,data_Sub09))
np.random.shuffle(data)

#Splitting the data into train and test data
data_train,data_test=train_test_split(data,test_size=0.25)

#Creating graph. Note that size_l wasnt constant for all runs.
mse_l=[]
size_l=range(20,801,20)

for i in size_l:
    clf = MLPClassifier(hidden_layer_sizes=(i),max_iter=10000,tol=1e-8)
    clf.fit(data_train[:,:-3],data_train[:,-3:])
    mse_l.append(clf.score(data_test[:,:-3],data_test[:,-3:]))

plt.plot(size_l,mse_l)
plt.show()
