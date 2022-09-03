
import pandas as pd
# import matplotlib.pyplot as plt
# import numpy as np


# part-1: Data pre-processing
dataset=pd.read_csv('Churn_Modelling.csv')

X=dataset.iloc[:,3:13].values
Y=dataset.iloc[:,-1].values

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
country_x=LabelEncoder()
X[:,1]=country_x.fit_transform(X[:,1])

gender_x=LabelEncoder()
X[:,2]=gender_x.fit_transform(X[:,2])

#creating dummy variable for country
from sklearn.compose import ColumnTransformer
ct=ColumnTransformer([("Country", OneHotEncoder(),[1])], remainder="passthrough")
X=ct.fit_transform(X)
X=X[:,1:]

# SPLITTING dataset into train and test set
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=0)


# FEATURE SCALING
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)



# PART 2: Making of ARTIFICAL nEURAL NETWORK (ANN)
import keras
import tensorflow as tf

from keras.models import Sequential
from keras.layers import Dense

# intialize ANN
classifier= Sequential()

# Add layers to ANN starting with input layer followed by hidden layer
classifier.add(Dense(6,activation='relu',kernel_initializer='uniform',input_dim=11))

# Add second hidden layer and so on
classifier.add(Dense(6,activation='relu',kernel_initializer='uniform'))

#add output layer
classifier.add(Dense(1,activation='sigmoid',kernel_initializer='uniform'))

#compile ANN (Stochastic gradient descent-back propagation)
classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

#fitting ANN to training set
classifier.fit(X_train,Y_train,batch_size=10,epochs=100)

#prediction on test
y_pred=classifier.predict(X_test)
y_pred=(y_pred > 0.5)
#create confusion matrix
from sklearn.metrics import confusion_matrix

cm=confusion_matrix(Y_test,y_pred)

print(cm)

acc=(cm[0,0]+cm[1,1])/2000
print(acc)



