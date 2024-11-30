import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Step 1: Load the dataset from CSV
data = pd.read_csv('../breast_implant_data_1.csv')

# Step 2: Define the independent variables (volumen_pre, proyeccion) and dependent variables (reborde, inframamaria)
X = data[['volumen_pre', 'proyeccion']]  # Independent variables
y = data[['reborde', 'inframamaria']]  # Dependent variables

# Step 3: Polynomial transformation
degree = 4  # You can adjust the degree of the polynomial here
poly = PolynomialFeatures(degree=degree)
X_poly = poly.fit_transform(X)

# Step 4: Fit the polynomial regression model
poly_model = LinearRegression()
poly_model.fit(X_poly, y)

# Step 5: Make predictions
y_pred = poly_model.predict(X_poly)

# Step 6: Evaluate the model performance
mse = mean_squared_error(y, y_pred)
r2 = r2_score(y, y_pred)

print(f"Polynomial Regression Model (Degree {degree})")
print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")

# Step 7: Plotting the results (for 2D visualization, we'll use scatter plots for each feature)

# Plot for 'reborde'
plt.figure(figsize=(14, 6))

# 'volumen_pre' vs 'reborde'
plt.subplot(1, 2, 1)
plt.scatter(data['volumen_pre'], data['reborde'], color='blue', label='Actual reborde')
plt.scatter(data['volumen_pre'], y_pred[:, 0], color='red', label='Fitted reborde (Polynomial)')
plt.title('Reborde vs Volumen Pre (Polynomial Regression)')
plt.xlabel('Volumen Pre')
plt.ylabel('Reborde')
plt.legend()

# 'proyeccion' vs 'reborde'
plt.subplot(1, 2, 2)
plt.scatter(data['proyeccion'], data['reborde'], color='blue', label='Actual reborde')
plt.scatter(data['proyeccion'], y_pred[:, 0], color='red', label='Fitted reborde (Polynomial)')
plt.title('Reborde vs Proyeccion (Polynomial Regression)')
plt.xlabel('Proyeccion')
plt.ylabel('Reborde')
plt.legend()

plt.tight_layout()
plt.show()

# Plot for 'inframamaria'
plt.figure(figsize=(14, 6))

# 'volumen_pre' vs 'inframamaria'
plt.subplot(1, 2, 1)
plt.scatter(data['volumen_pre'], data['inframamaria'], color='green', label='Actual inframamaria')
plt.scatter(data['volumen_pre'], y_pred[:, 1], color='orange', label='Fitted inframamaria (Polynomial)')
plt.title('Inframamaria vs Volumen Pre (Polynomial Regression)')
plt.xlabel('Volumen Pre')
plt.ylabel('Inframamaria')
plt.legend()

# 'proyeccion' vs 'inframamaria'
plt.subplot(1, 2, 2)
plt.scatter(data['proyeccion'], data['inframamaria'], color='green', label='Actual inframamaria')
plt.scatter(data['proyeccion'], y_pred[:, 1], color='orange', label='Fitted inframamaria (Polynomial)')
plt.title('Inframamaria vs Proyeccion (Polynomial Regression)')
plt.xlabel('Proyeccion')
plt.ylabel('Inframamaria')
plt.legend()

plt.tight_layout()
plt.show()
