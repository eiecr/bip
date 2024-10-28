import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from pygam import LinearGAM, s
from sklearn.metrics import mean_squared_error, r2_score

# Step 1: Load the dataset
data = pd.read_csv('breast_implant_data_1.csv')

# Step 2: Remove rows where 'proyeccion_medida' is NaN
data = data.dropna(subset=['proyeccion_medida'])

# Step 3: Define independent (X) and dependent (y) variables
X = data[['volumen_pre', 'proyeccion', 'reborde', 'inframamaria']]
y = data[['base_implante_colocado', 'volumen_colocado', 'proyeccion_medida']]

# Step 4: Scale the independent variables (X)
scaler_X = StandardScaler()
X_scaled = scaler_X.fit_transform(X)

# Step 5: Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Step 6: Fit separate GAMs for each target variable
models = {}
predictions = {}

for target in y.columns:
    # Initialize and fit GAM for each target variable
    gam = LinearGAM(s(0) + s(1) + s(2) + s(3))  # One smooth function per feature
    gam.fit(X_train, y_train[target])

    # Store the model and predictions for later evaluation
    models[target] = gam
    predictions[target] = gam.predict(X_test)

# Step 7: Evaluate the models and calculate metrics
mse_scores = {}
r2_scores = {}

for target in y.columns:
    mse_scores[target] = mean_squared_error(y_test[target], predictions[target])
    r2_scores[target] = r2_score(y_test[target], predictions[target])

print("Mean Squared Error for each target variable:", mse_scores)
print("R² score for each target variable:", r2_scores)

# Step 8: Plot Actual vs Predicted for each target variable
plt.figure(figsize=(18, 5))

for i, target in enumerate(y.columns):
    plt.subplot(1, len(y.columns), i + 1)
    plt.scatter(y_test[target], predictions[target], alpha=0.5)
    plt.plot([y_test[target].min(), y_test[target].max()], [y_test[target].min(), y_test[target].max()],
             'r--')  # 45-degree line
    plt.xlabel(f"Actual {target}")
    plt.ylabel(f"Predicted {target}")
    plt.title(f"Actual vs Predicted for {target}")

plt.tight_layout()
plt.show()

# Step 9: Plot GAM smooth functions for each target variable
for target in y.columns:
    gam = models[target]
    plt.figure(figsize=(14, 6))
    plt.suptitle(f'Smooth Functions for Target: {target}')

    for i in range(X.shape[1]):  # Iterate over each feature by index
        XX = gam.generate_X_grid(term=i)
        plt.subplot(1, X.shape[1], i + 1)
        plt.plot(XX[:, i], gam.partial_dependence(term=i, X=XX))
        plt.xlabel(X.columns[i])
        plt.ylabel(f'Smooth effect on {target}')
        plt.title(f'Smooth function for {X.columns[i]}')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()
