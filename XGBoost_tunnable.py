## needs XGBoost Library (libxgboost.dylib)
## mac:
#  brew install libomp or pip install xgboost --no-binary :all:
## brew link --overwrite libomp --force
import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV, cross_val_score
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

# Step 1: Load the dataset from CSV
data = pd.read_csv('breast_implant_data_1.csv')

# Step 2: Define the independent variables (volumen_pre, proyeccion) and dependent variables (reborde, inframamaria)
X = data[['volumen_pre', 'proyeccion']]  # Independent variables
y_reborde = data['reborde']  # Dependent variable: reborde
y_inframamaria = data['inframamaria']  # Dependent variable: inframamaria

# Step 3: Hyperparameter tuning using RandomizedSearchCV for XGBoost

# Define parameter grid for XGBoost
param_distributions = {
    'n_estimators': [50, 100, 200, 400],
    'max_depth': [3, 5, 10, 20],
    'learning_rate': [0.01, 0.05, 0.1],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'gamma': [0, 0.1, 0.3, 0.5]
}

# Initialize XGBoost Regressor
xgb_model = xgb.XGBRegressor(random_state=42)

# RandomizedSearchCV for 'reborde'
random_search_reborde = RandomizedSearchCV(
    estimator=xgb_model,
    param_distributions=param_distributions,
    n_iter=10,  # Number of random combinations to try
    cv=5,
    n_jobs=-1,
    verbose=2,
    scoring='neg_mean_squared_error',
    random_state=42
)
random_search_reborde.fit(X, y_reborde)

# Best parameters for 'reborde'
best_params_reborde = random_search_reborde.best_params_
print(f"Best parameters for 'reborde': {best_params_reborde}")

# RandomizedSearchCV for 'inframamaria'
random_search_inframamaria = RandomizedSearchCV(
    estimator=xgb_model,
    param_distributions=param_distributions,
    n_iter=10,  # Number of random combinations to try
    cv=5,
    n_jobs=-1,
    verbose=2,
    scoring='neg_mean_squared_error',
    random_state=42
)
random_search_inframamaria.fit(X, y_inframamaria)

# Best parameters for 'inframamaria'
best_params_inframamaria = random_search_inframamaria.best_params_
print(f"Best parameters for 'inframamaria': {best_params_inframamaria}")

# Step 4: Use cross-validation to evaluate the model with the best parameters

# XGBoost with best parameters for 'reborde'
xgb_best_reborde = xgb.XGBRegressor(**best_params_reborde, random_state=42)

# Perform cross-validation for 'reborde'
cv_scores_reborde = cross_val_score(xgb_best_reborde, X, y_reborde, cv=5, scoring='neg_mean_squared_error')
print(f"Cross-validated MSE for 'reborde': {np.mean(-cv_scores_reborde)}")

# XGBoost with best parameters for 'inframamaria'
xgb_best_inframamaria = xgb.XGBRegressor(**best_params_inframamaria, random_state=42)

# Perform cross-validation for 'inframamaria'
cv_scores_inframamaria = cross_val_score(xgb_best_inframamaria, X, y_inframamaria, cv=5, scoring='neg_mean_squared_error')
print(f"Cross-validated MSE for 'inframamaria': {np.mean(-cv_scores_inframamaria)}")

# Step 5: Fit the model and evaluate feature importance

# Fit the model for 'reborde'
xgb_best_reborde.fit(X, y_reborde)

# Get feature importances for 'reborde'
importances_reborde = xgb_best_reborde.feature_importances_
print("Feature importances for 'reborde':")
for feature, importance in zip(X.columns, importances_reborde):
    print(f"{feature}: {importance:.4f}")

# Plot feature importances for 'reborde'
plt.bar(X.columns, importances_reborde)
plt.title("Feature Importance for 'Reborde'")
plt.show()

# Fit the model for 'inframamaria'
xgb_best_inframamaria.fit(X, y_inframamaria)

# Get feature importances for 'inframamaria'
importances_inframamaria = xgb_best_inframamaria.feature_importances_
print("Feature importances for 'inframamaria':")
for feature, importance in zip(X.columns, importances_inframamaria):
    print(f"{feature}: {importance:.4f}")

# Plot feature importances for 'inframamaria'
plt.bar(X.columns, importances_inframamaria)
plt.title("Feature Importance for 'Inframamaria'")
plt.show()

# Step 6: Make predictions with the tuned model
y_pred_reborde = xgb_best_reborde.predict(X)
y_pred_inframamaria = xgb_best_inframamaria.predict(X)

# Evaluate the final model performance
mse_reborde = mean_squared_error(y_reborde, y_pred_reborde)
r2_reborde = r2_score(y_reborde, y_pred_reborde)
print(f"Reborde - Final Mean Squared Error: {mse_reborde}")
print(f"Reborde - Final R² Score: {r2_reborde}")

mse_inframamaria = mean_squared_error(y_inframamaria, y_pred_inframamaria)
r2_inframamaria = r2_score(y_inframamaria, y_pred_inframamaria)
print(f"Inframamaria - Final Mean Squared Error: {mse_inframamaria}")
print(f"Inframamaria - Final R² Score: {r2_inframamaria}")


# Step 7: Plotting the results

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
