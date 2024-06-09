import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

sonar_data = pd.read_csv('Rock-vs-Mine-Prediction\sonar_data.csv', header=None)

sonar_data.head()

sonar_data[60].value_counts()

sonar_data.groupby(60).mean()

# separating data and Labels
X = sonar_data.drop(columns=60, axis=1)
Y = sonar_data[60]

print(X)
print(Y)

#training and test data

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.1, stratify=Y, random_state=1)


print(X.shape, X_train.shape, X_test.shape)

print(X_train)
print(Y_train)

#Model Training --> Logistic Regression

model = LogisticRegression()

#training the Logistic Regression model with training data
model.fit(X_train, Y_train)

#accuracy on training data
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train) 

print('Accuracy on training data : ', training_data_accuracy) #printing accuracy on training

#accuracy on test data
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test) 

print('Accuracy on test data : ', test_data_accuracy)  #printing accuracy on test

#Making a Predictive System

input_data = (0.0408,0.0653,0.0397,0.0604,0.0496,0.1817,0.1178,0.1024,0.0583,0.2176,0.2459,0.3332,0.3087,0.2613,0.3232,0.3731,0.4203,0.5364,0.7062,0.8196,0.8835,0.8299,0.7609,0.7605,0.8367,0.8905,0.7652,0.5897,0.3037,0.0823,0.2787,0.7241,0.8032,0.8050,0.7676,0.7468,0.6253,0.1730,0.2916,0.5003,0.5220,0.4824,0.4004,0.3877,0.1651,0.0442,0.0663,0.0418,0.0475,0.0235,0.0066,0.0062,0.0129,0.0184,0.0069,0.0198,0.0199,0.0102,0.0070,0.0055)

# changing the input_data to a numpy array
input_data_as_numpy_array = np.asarray(input_data)

# reshape the np array as we are predicting for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction = model.predict(input_data_reshaped)
print(prediction)

if (prediction[0]=='R'):
  print('The object is a Rock')
else:
  print('The object is a mine')
