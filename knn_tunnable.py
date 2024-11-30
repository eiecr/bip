from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Step 1: Load the dataset from CSV
data = pd.read_csv('breast_implant_data_1.csv')

data = data.dropna(subset=['proyeccion_medida'])

# Step 2: Define the independent variables (volumen_pre, proyeccion) and dependent variables (reborde, inframamaria)
X = data[['volumen_pre', 'proyeccion']]  # Independent variables
y_reborde = data['volumen_colocado']  # Dependent variable: reborde
y_inframamaria = data['proyeccion_medida']  # Dependent variable: inframamaria

# Step 3: Scale the independent variables using StandardScaler
scaler_X = StandardScaler()  # Initialize scaler
X_scaled = scaler_X.fit_transform(X)  # Scale independent variables

# Step 4: Hyperparameter tuning using GridSearchCV for KNN

# Define parameter grid for KNN
param_grid = {
    'n_neighbors': [3, 5, 7, 10],
    'weights': ['uniform', 'distance'],  # Uniform weights or distance-weighted neighbors
    'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
}

# Initialize KNeighborsRegressor
knn = KNeighborsRegressor()

# GridSearchCV for 'reborde'
grid_search_reborde = GridSearchCV(estimator=knn, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2, scoring='neg_mean_squared_error')
grid_search_reborde.fit(X_scaled, y_reborde)

# Best parameters for 'reborde'
best_params_reborde = grid_search_reborde.best_params_
print(f"Best parameters for 'reborde': {best_params_reborde}")

# GridSearchCV for 'inframamaria'
grid_search_inframamaria = GridSearchCV(estimator=knn, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2, scoring='neg_mean_squared_error')
grid_search_inframamaria.fit(X_scaled, y_inframamaria)

# Best parameters for 'inframamaria'
best_params_inframamaria = grid_search_inframamaria.best_params_
print(f"Best parameters for 'inframamaria': {best_params_inframamaria}")

# Step 5: Use cross-validation to evaluate the model with the best parameters

# KNN with best parameters for 'reborde'
knn_best_reborde = KNeighborsRegressor(**best_params_reborde)

# Perform cross-validation for 'reborde'
cv_scores_reborde = cross_val_score(knn_best_reborde, X_scaled, y_reborde, cv=5, scoring='neg_mean_squared_error')
print(f"Cross-validated MSE for 'reborde': {np.mean(-cv_scores_reborde)}")

# KNN with best parameters for 'inframamaria'
knn_best_inframamaria = KNeighborsRegressor(**best_params_inframamaria)

# Perform cross-validation for 'inframamaria'
cv_scores_inframamaria = cross_val_score(knn_best_inframamaria, X_scaled, y_inframamaria, cv=5, scoring='neg_mean_squared_error')
print(f"Cross-validated MSE for 'inframamaria': {np.mean(-cv_scores_inframamaria)}")

# Step 6: Fit the model and make predictions with the tuned model
knn_best_reborde.fit(X_scaled, y_reborde)
knn_best_inframamaria.fit(X_scaled, y_inframamaria)

# Predict using the tuned model
y_pred_reborde = knn_best_reborde.predict(X_scaled)
y_pred_inframamaria = knn_best_inframamaria.predict(X_scaled)

# Step 7: Evaluate the final model performance
mse_reborde = mean_squared_error(y_reborde, y_pred_reborde)
r2_reborde = r2_score(y_reborde, y_pred_reborde)
print(f"Reborde - Final Mean Squared Error: {mse_reborde}")
print(f"Reborde - Final R² Score: {r2_reborde}")

mse_inframamaria = mean_squared_error(y_inframamaria, y_pred_inframamaria)
r2_inframamaria = r2_score(y_inframamaria, y_pred_inframamaria)
print(f"Inframamaria - Final Mean Squared Error: {mse_inframamaria}")
print(f"Inframamaria - Final R² Score: {r2_inframamaria}")

# Step 8: Plotting (Optional)
plt.scatter(X['volumen_pre'], y_reborde, color='blue', label='Actual reborde')
plt.scatter(X['volumen_pre'], y_pred_reborde, color='red', label='Predicted reborde (KNN)')
plt.title('Reborde vs Volumen Pre (KNN)')
plt.xlabel('Volumen Pre')
plt.ylabel('Reborde')
plt.legend()
plt.show()

plt.scatter(X['volumen_pre'], y_inframamaria, color='green', label='Actual inframamaria')
plt.scatter(X['volumen_pre'], y_pred_inframamaria, color='orange', label='Predicted inframamaria (KNN)')
plt.title('Inframamaria vs Volumen Pre (KNN)')
plt.xlabel('Volumen Pre')
plt.ylabel('Inframamaria')
plt.legend()
plt.show()

