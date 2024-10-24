import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Step 1: Load the dataset from CSV
data = pd.read_csv('breast_implant_data_1.csv')

# Step 2: Define the independent variables (volumen_pre, proyeccion) and dependent variables (reborde, inframamaria)
X = data[['volumen_pre', 'proyeccion']]  # Independent variables
y = data[['reborde', 'inframamaria']]  # Dependent variables

# Step 3: Fit the Random Forest Regressor
rf_model_reborde = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model_inframamaria = RandomForestRegressor(n_estimators=100, random_state=42)

# Fit the model for 'reborde'
rf_model_reborde.fit(X, y['reborde'])

# Fit the model for 'inframamaria'
rf_model_inframamaria.fit(X, y['inframamaria'])

# Step 4: Make predictions
y_pred_reborde = rf_model_reborde.predict(X)
y_pred_inframamaria = rf_model_inframamaria.predict(X)

# Step 5: Evaluate the model performance
mse_reborde = mean_squared_error(y['reborde'], y_pred_reborde)
r2_reborde = r2_score(y['reborde'], y_pred_reborde)
print(f"Reborde - Mean Squared Error: {mse_reborde}")
print(f"Reborde - R² Score: {r2_reborde}")

mse_inframamaria = mean_squared_error(y['inframamaria'], y_pred_inframamaria)
r2_inframamaria = r2_score(y['inframamaria'], y_pred_inframamaria)
print(f"Inframamaria - Mean Squared Error: {mse_inframamaria}")
print(f"Inframamaria - R² Score: {r2_inframamaria}")

# Step 6: Plotting the results

# Plot for 'reborde'
plt.figure(figsize=(14, 6))

# 'volumen_pre' vs 'reborde'
plt.subplot(1, 2, 1)
plt.scatter(data['volumen_pre'], data['reborde'], color='blue', label='Actual reborde')
plt.scatter(data['volumen_pre'], y_pred_reborde, color='red', label='Predicted reborde (Random Forest)')
plt.title('Reborde vs Volumen Pre (Random Forest)')
plt.xlabel('Volumen Pre')
plt.ylabel('Reborde')
plt.legend()

# 'proyeccion' vs 'reborde'
plt.subplot(1, 2, 2)
plt.scatter(data['proyeccion'], data['reborde'], color='blue', label='Actual reborde')
plt.scatter(data['proyeccion'], y_pred_reborde, color='red', label='Predicted reborde (Random Forest)')
plt.title('Reborde vs Proyeccion (Random Forest)')
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
plt.scatter(data['volumen_pre'], y_pred_inframamaria, color='orange', label='Predicted inframamaria (Random Forest)')
plt.title('Inframamaria vs Volumen Pre (Random Forest)')
plt.xlabel('Volumen Pre')
plt.ylabel('Inframamaria')
plt.legend()

# 'proyeccion' vs 'inframamaria'
plt.subplot(1, 2, 2)
plt.scatter(data['proyeccion'], data['inframamaria'], color='green', label='Actual inframamaria')
plt.scatter(data['proyeccion'], y_pred_inframamaria, color='orange', label='Predicted inframamaria (Random Forest)')
plt.title('Inframamaria vs Proyeccion (Random Forest)')
plt.xlabel('Proyeccion')
plt.ylabel('Inframamaria')
plt.legend()

plt.tight_layout()
plt.show()
