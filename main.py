import numpy as np
from sklearn.linear_model import LinearRegression

# Training data
X = np.array([
    [2, 6, 40],
    [3, 7, 50],
    [4, 6, 60],
    [5, 8, 79],
    [6, 7, 85]
])
y = np.array([50, 60, 65, 80, 85])

# Train model
model = LinearRegression()
model.fit(X, y)

# Take user input
study = float(input("Enter study hours: "))
sleep = float(input("Enter sleep hours: "))
attendance = float(input("Enter attendance: "))

# Prediction
marks = model.predict([[study, sleep, attendance]])[0]

print("Predicted Marks:", marks)

# Logic layer
if marks < 50:
    print("Result: Fail risk. Increase study hours.")
elif marks < 70:
    print("Result: Average. Improve consistency.")
else:
    print("Result: Good performance.")