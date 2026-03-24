import numpy as np 
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
x = np.array([1,2,3,4,5]).reshape(-1,1)
y = np.array([10,20,30,40,50])

model = LinearRegression()

model.fit(x,y)

m =model.coef_[0]
b =model.intercept_

print("M (Slope)-",m)
print("B Bias-",b)

x_new =np.array([[6],[7],[8]])
y_pred =model.predict(x_new)

print("Prediction is for 6,7,8 is -",y_pred)


plt.scatter(x,y, color='blue')
plt.plot(x,model.predict(x),color='green')
plt.show()