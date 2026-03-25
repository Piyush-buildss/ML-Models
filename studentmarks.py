import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 
from sklearn.linear_model import LinearRegression
data =pd.read_csv("Student_Marks.csv")

print(data.head())
df=pd.DataFrame(data)
x =df[["number_courses","time_study"]]
y =df[["Marks"]]
print(x[["number_courses"]])

model= LinearRegression()

model.fit(x,y)
x_new=[[4,5.455]]
m =model.coef_[0]
b =model.intercept_
print("slope:-",m)
print("Bias :-",b)

y_pred =model.predict(x_new)
print("Marks",y_pred)