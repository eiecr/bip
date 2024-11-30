import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


# Step 1: Load your data from the CSV file
data = pd.read_csv('../breast_implant_data_1.csv')

# Step 2: Define the independent variables (volumen_pre, proyeccion) and dependent variables (reborde, inframamaria)
X = data[['volumen_pre', 'proyeccion']]  # Independent variables
y = data[['reborde', 'inframamaria']]  # Dependent variables

# Step 3: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Train the multivariate linear regression model using scikit-learn
model = LinearRegression()
model.fit(X_train, y_train)

# Step 5: Check model coefficients and intercept
print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)

# Step 6: Make predictions on the test set
y_pred = model.predict(X_test)

# Step 7: Evaluate the model
# Evaluate for both dependent variables
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("Mean Squared Error:", mse)
print("R-squared:", r2)

# Step 8: Detailed statistical analysis using statsmodels (for both dependent variables separately)

# Model for reborde (mammary rim)
y_train_reborde = y_train['reborde']
X_train_sm = sm.add_constant(X_train)  # Adds a constant term (intercept) to the predictors
ols_model_reborde = sm.OLS(y_train_reborde, X_train_sm)
results_reborde = ols_model_reborde.fit()

print("Reborde Model Summary:")
print(results_reborde.summary())

# Model for inframamaria
y_train_inframamaria = y_train['inframamaria']
ols_model_inframamaria = sm.OLS(y_train_inframamaria, X_train_sm)
results_inframamaria = ols_model_inframamaria.fit()

print("\nInframamaria Model Summary:")
print(results_inframamaria.summary())

# Step 9: Plotting the regression results

# Plot for reborde (mammary rim)
plt.figure(figsize=(12, 5))

# Plot scatter plot for volumen_pre vs reborde
plt.subplot(1, 2, 1)
plt.scatter(X_train['volumen_pre'], y_train_reborde, color='blue', label='Actual reborde')
plt.plot(X_train['volumen_pre'], results_reborde.predict(X_train_sm), color='red', label='Fitted line')
plt.xlabel('Volumen Pre')
plt.ylabel('Reborde')
plt.title('Reborde vs Volumen Pre')
plt.legend()

# Plot scatter plot for proyeccion vs reborde
plt.subplot(1, 2, 2)
plt.scatter(X_train['proyeccion'], y_train_reborde, color='blue', label='Actual reborde')
plt.plot(X_train['proyeccion'], results_reborde.predict(X_train_sm), color='red', label='Fitted line')
plt.xlabel('Proyeccion')
plt.ylabel('Reborde')
plt.title('Reborde vs Proyeccion')
plt.legend()

plt.tight_layout()
plt.show()

# Plot for inframamaria (inframammary fold)
plt.figure(figsize=(12, 5))

# Plot scatter plot for volumen_pre vs inframamaria
plt.subplot(1, 2, 1)
plt.scatter(X_train['volumen_pre'], y_train_inframamaria, color='green', label='Actual inframamaria')
plt.plot(X_train['volumen_pre'], results_inframamaria.predict(X_train_sm), color='orange', label='Fitted line')
plt.xlabel('Volumen Pre')
plt.ylabel('Inframamaria')
plt.title('Inframamaria vs Volumen Pre')
plt.legend()

# Plot scatter plot for proyeccion vs inframamaria
plt.subplot(1, 2, 2)
plt.scatter(X_train['proyeccion'], y_train_inframamaria, color='green', label='Actual inframamaria')
plt.plot(X_train['proyeccion'], results_inframamaria.predict(X_train_sm), color='orange', label='Fitted line')
plt.xlabel('Proyeccion')
plt.ylabel('Inframamaria')
plt.title('Inframamaria vs Proyeccion')
plt.legend()

plt.tight_layout()
plt.show()
