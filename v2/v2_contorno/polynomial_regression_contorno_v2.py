import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Step 1: Load the dataset from CSV
data = pd.read_csv('datos_contorno.csv')

# Step 2: Define the independent variables (contorno_iz_6_mes) and dependent variables (avg volume 6 months after surgery)
X = data[['contorno_iz_6_mes']]  # Independent variables
y = data[['volume_avg']]  # Dependent variables

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

# Plot for 'infra'
plt.figure(figsize=(14, 6))
# infra_6_mes vs volume_avg
# 'volumen_pre' vs 'reborde'
plt.subplot(1, 2, 1)
plt.scatter(data['contorno_iz_6_mes'], data['volume_avg'], color='blue', label='Actual volume')
plt.scatter(data['contorno_iz_6_mes'], y_pred[:, 0], color='red', label='Fitted volume_avg (Polynomial)')
plt.title('Volumen (avg) vs contorno_iz_6_mes (Polynomial Regression)')
plt.xlabel('contorno_iz_6_mes')
plt.ylabel('Volume (avg)')
plt.legend()

plt.tight_layout()
plt.show()