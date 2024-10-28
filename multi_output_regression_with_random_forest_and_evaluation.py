import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Load the dataset
data = pd.read_csv('breast_implant_data_1.csv')

# Remove rows where 'proyeccion_medida' is NaN
data = data.dropna(subset=['proyeccion_medida'])

# Define independent (X) and dependent (y) variables
X = data[['volumen_pre', 'proyeccion', 'reborde', 'inframamaria']]
y = data[['base_implante_colocado', 'volumen_colocado', 'proyeccion_medida']]

# Scale the independent variables (X)
scaler_X = StandardScaler()
X_scaled = scaler_X.fit_transform(X)

# Optional: Split data into training and testing sets for more robust evaluation
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Initialize Multi-Output Regressor with Random Forest as the base estimator
model = MultiOutputRegressor(RandomForestRegressor(random_state=42))
model.fit(X_train, y_train)

# Predict on the test data
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred, multioutput='raw_values')
r2 = r2_score(y_test, y_pred, multioutput='raw_values')

print("Mean Squared Error for each target variable:", mse)
print("R² score for each target variable:", r2)

# Plotting Actual vs Predicted for each target variable
target_columns = y.columns  # Column names of the target variables
plt.figure(figsize=(18, 5))

for i, target in enumerate(target_columns):
    plt.subplot(1, len(target_columns), i + 1)
    plt.scatter(y_test[target], y_pred[:, i], alpha=0.5)
    plt.plot([y_test[target].min(), y_test[target].max()], [y_test[target].min(), y_test[target].max()], 'r--')  # 45-degree line
    plt.xlabel(f"Actual {target}")
    plt.ylabel(f"Predicted {target}")
    plt.title(f"Actual vs Predicted for {target}")

plt.tight_layout()
plt.show()
