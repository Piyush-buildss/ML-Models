import numpy as np 
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error ,r2_score

data = {
    "area": [1000,1500,1800,2400,3000,1200,2000,2200,2700,3200],
    "bedrooms": [2,3,3,4,4,2,3,3,4,5],
    "age": [20,15,10,5,2,18,8,6,3,1],
    "price": [200000,300000,350000,500000,650000,220000,400000,450000,600000,700000]
}

df =pd.DataFrame(data)

X =df[["area","bedrooms","age"]]
y= df["price"]

X_train ,X_test ,y_train ,y_test =train_test_split(X,y,test_size=0.2,random_state=42)

model =LinearRegression()
model.fit(X_train,y_train)

y_pred =model.predict(X_test)

print("Mse:",mean_squared_error(y_test,y_pred))
print("R2:",r2_score(y_test,y_pred))


print("Coefficients:",model.coef_)
print("Intercept:",model.intercept_)

print("Actual :",y_test.values)
print("Predicted:",y_pred)

new_house = [[2560,6,3]]
prediction = model.predict(new_house)

print("Predicted Price of new House -",prediction)