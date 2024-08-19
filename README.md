# Developing a Neural Network Regression Model

## AIM

To develop a neural network regression model for the given dataset.

## THEORY

Explain the problem statement

## Neural Network Model

Include the neural network model diagram.
![image](https://github.com/user-attachments/assets/607daa30-d76b-4408-9dda-b4228f5077e1)


## DESIGN STEPS

### STEP 1:

Loading the dataset

### STEP 2:

Split the dataset into training and testing

### STEP 3:

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4:

Build the Neural Network Model and compile the model.

### STEP 5:

Train the model with the training data.

### STEP 6:

Plot the performance plot

### STEP 7:

Evaluate the model with the testing data.

## PROGRAM
### Name:E.Jeevitha
### Register Number:212222230054
```python

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from google.colab  import auth
import gspread
from google.auth import default
import pandas as pd
auth.authenticate_user()
creds,_=default()
gc = gspread.authorize(creds)
worksheet =  gc.open('data').sheet1
data = worksheet.get_all_values()
df=pd.DataFrame(data[1:], columns=data[0])
df=df.astype({'X':'int'})
df=df.astype({'Y':'int'})
df.head()
X = df[['X']].values
Y = df[['Y']].values
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, random_state=33)
Scaler = MinMaxScaler()
Scaler.fit(X_train)
X_train1 = Scaler.transform(X_train)
ai_brain = Sequential([
    Dense(8,activation = 'relu'),
    Dense(10,activation = 'relu'),
    Dense(1)
])
ai_brain.compile(optimizer= 'rmsprop',loss = 'mse')
ai_brain.fit(X_train1,Y_train,epochs= 2000)
loss_df = pd.DataFrame(ai_brain.history.history)
loss_df.plot()
X_test1 = Scaler.transform(X_test)
ai_brain.evaluate(X_test1,Y_test)
X_n1 = [[7]]
X_n1_1 = Scaler.transform(X_n1)
ai_brain.predict(X_n1_1 )

```
## Dataset Information

![image](https://github.com/user-attachments/assets/a6829eb3-f318-4c4b-9010-2c15a83e2a46)


## OUTPUT

### Training Loss Vs Iteration Plot

![image](https://github.com/user-attachments/assets/9fc7f3ce-ec80-4e96-b5f5-97f5f616a0b9)


### Test Data Root Mean Squared Error

![image](https://github.com/user-attachments/assets/ea9d1ac7-7bfb-4bb4-bbfd-0c04a814a1a8)


### New Sample Data Prediction

![image](https://github.com/user-attachments/assets/c5497ca1-6a8a-4de6-878d-20f44407f275)


## RESULT

Thus a Neural network for Regression model is Implemented.
