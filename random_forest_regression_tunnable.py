import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# Step 1: Load the dataset from CSV
data = pd.read_csv('breast_implant_data_1.csv')

# Step 2: Define the independent variables (volumen_pre, proyeccion) and dependent variables (reborde, inframamaria)
X = data[['volumen_pre', 'proyeccion']]  # Independent variables
y_reborde = data[['reborde']]  # Dependent variable: reborde
y_inframamaria = data[['inframamaria']]  # Dependent variable: inframamaria

# Step 3: Initialize scalers
scaler_X = StandardScaler()  # Scaler for independent variables
scaler_y_reborde = StandardScaler()  # Scaler for reborde
scaler_y_inframamaria = StandardScaler()  # Scaler for inframamaria

# Step 4: Fit and transform the independent and dependent variables
X_scaled = scaler_X.fit_transform(X)
y_reborde_scaled = scaler_y_reborde.fit_transform(y_reborde)
y_inframamaria_scaled = scaler_y_inframamaria.fit_transform(y_inframamaria)


# Step 5: Hyperparameter tuning using GridSearchCV
param_grid = {
    'n_estimators': [50, 100, 200],  # Reduced range of n_estimators
    'max_depth': [None, 10, 20],     # Reduced range of max_depth
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': [2, 'sqrt']      # For two features, keep it simple
}

# Initialize RandomForestRegressor
rf = RandomForestRegressor(random_state=42)

# GridSearchCV for reborde
grid_search_reborde = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2, scoring='neg_mean_squared_error')
grid_search_reborde.fit(X_scaled, y_reborde_scaled.ravel())

# Best parameters for 'reborde'
best_params_reborde = grid_search_reborde.best_params_
print(f"Best parameters for 'reborde': {best_params_reborde}")

# GridSearchCV for inframamaria
grid_search_inframamaria = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2, scoring='neg_mean_squared_error')
grid_search_inframamaria.fit(X_scaled, y_inframamaria_scaled.ravel())

# Best parameters for 'inframamaria'
best_params_inframamaria = grid_search_inframamaria.best_params_
print(f"Best parameters for 'inframamaria': {best_params_inframamaria}")

# Step 6: Use cross-validation to evaluate the model with the best parameters

# RandomForestRegressor with best parameters for 'reborde'
rf_best_reborde = RandomForestRegressor(**best_params_reborde, random_state=42)

# Perform cross-validation for 'reborde'
cv_scores_reborde = cross_val_score(rf_best_reborde, X_scaled, y_reborde_scaled.ravel(), cv=5, scoring='neg_mean_squared_error')
print(f"Cross-validated MSE for 'reborde': {np.mean(-cv_scores_reborde)}")

# RandomForestRegressor with best parameters for 'inframamaria'
rf_best_inframamaria = RandomForestRegressor(**best_params_inframamaria, random_state=42)

# Perform cross-validation for 'inframamaria'
cv_scores_inframamaria = cross_val_score(rf_best_inframamaria, X_scaled, y_inframamaria_scaled.ravel(), cv=5, scoring='neg_mean_squared_error')
print(f"Cross-validated MSE for 'inframamaria': {np.mean(-cv_scores_inframamaria)}")

# Step 7: Fit the model and evaluate feature importance

# Fit the model for 'reborde'
rf_best_reborde.fit(X_scaled, y_reborde_scaled.ravel())

# Get feature importances for 'reborde'
importances_reborde = rf_best_reborde.feature_importances_
print("Feature importances for 'reborde':")
for feature, importance in zip(X.columns, importances_reborde):
    print(f"{feature}: {importance:.4f}")

# Plot feature importances for 'reborde'
plt.bar(X.columns, importances_reborde)
plt.title("Feature Importance for 'Reborde'")
plt.show()

# Fit the model for 'inframamaria'
rf_best_inframamaria.fit(X_scaled, y_inframamaria_scaled.ravel())

# Get feature importances for 'inframamaria'
importances_inframamaria = rf_best_inframamaria.feature_importances_
print("Feature importances for 'inframamaria':")
for feature, importance in zip(X.columns, importances_inframamaria):
    print(f"{feature}: {importance:.4f}")

# Plot feature importances for 'inframamaria'
plt.bar(X.columns, importances_inframamaria)
plt.title("Feature Importance for 'Inframamaria'")
plt.show()

# Step 8: Make predictions with the tuned model
y_pred_reborde_scaled = rf_best_reborde.predict(X_scaled)
y_pred_inframamaria_scaled = rf_best_inframamaria.predict(X_scaled)


# Inverse transform the predictions and actuals to their original scale
y_pred_reborde = np.nan_to_num(scaler_y_reborde.inverse_transform(y_pred_reborde_scaled.reshape(-1, 1)))
y_pred_inframamaria = np.nan_to_num(scaler_y_inframamaria.inverse_transform(y_pred_inframamaria_scaled.reshape(-1, 1)))
y_reborde = scaler_y_reborde.inverse_transform(y_reborde_scaled)
y_inframamaria = scaler_y_inframamaria.inverse_transform(y_inframamaria_scaled)

# Step 9: Evaluate the final model performance on the original scale
mse_reborde = mean_squared_error(y_reborde, y_pred_reborde)
r2_reborde = r2_score(y_reborde, y_pred_reborde)
print(f"Reborde - Final Mean Squared Error: {mse_reborde}")
print(f"Reborde - Final R² Score: {r2_reborde}")

mse_inframamaria = mean_squared_error(y_inframamaria, y_pred_inframamaria)
r2_inframamaria = r2_score(y_inframamaria, y_pred_inframamaria)
print(f"Inframamaria - Final Mean Squared Error: {mse_inframamaria}")
print(f"Inframamaria - Final R² Score: {r2_inframamaria}")

# Step 10: Plotting the results


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

