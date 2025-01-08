import numpy as np
import pandas as pd

# Preprocesamiento y partición de datos
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.preprocessing import StandardScaler

# Algoritmo de Random Forest
from sklearn.ensemble import RandomForestRegressor

# Métricas
from sklearn.metrics import mean_squared_error, r2_score

# Visualización
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------------------------------------------------------
# 1. Lectura de datos
# -----------------------------------------------------------------------------
data = pd.read_csv('datos_mezclados.csv')  # Ajusta el nombre de tu CSV

# -----------------------------------------------------------------------------
# 2. Filtrado de outliers en TODAS las variables predictoras
#    usando el criterio IQR (Interquartile Range).
#    Filtramos cada variable: 'infra_6_mes', 'contorno_avg', 'reborde_avg'.
# -----------------------------------------------------------------------------
def filtrar_outliers_iqr(df, columna, k=1.5):
    """Filtra outliers de la columna dada en df según IQR, usando factor k (default=1.5)."""
    Q1 = df[columna].quantile(0.25)
    Q3 = df[columna].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - k * IQR
    upper_bound = Q3 + k * IQR
    return df[(df[columna] >= lower_bound) & (df[columna] <= upper_bound)]

variables = ['infra_6_mes', 'contorno_avg', 'reborde_avg']
for var in variables:
    data = filtrar_outliers_iqr(data, var, k=1.5)

print("\n--- Filtrado de outliers (IQR) en las variables: infra_6_mes, contorno_avg, reborde_avg ---")
print(f"Registros restantes tras filtrar: {len(data)}")

# -----------------------------------------------------------------------------
# 3. Separación de características (X) y variable objetivo (y)
# -----------------------------------------------------------------------------
X = data[['infra_6_mes', 'contorno_avg', 'reborde_avg']].values
y = data['volumen'].values

# -----------------------------------------------------------------------------
# 4. División en entrenamiento y prueba (80% train - 20% test)
# -----------------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# -----------------------------------------------------------------------------
# 5. Estandarización
#    (Aunque Random Forest no depende fuertemente de la escala,
#     hacemos el escalado en caso de que luego combines con otras técnicas.)
# -----------------------------------------------------------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -----------------------------------------------------------------------------
# 6. Definir la rejilla de hiperparámetros
#    Ajusta estos valores según tu dataset y capacidad de cómputo.
# -----------------------------------------------------------------------------
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 5, 10, 20],
    'min_samples_split': [2, 5, 10],
    'max_features': [1, 2, 3],        # para 3 variables totales
    'bootstrap': [True, False]
}

# -----------------------------------------------------------------------------
# 7. Configurar el RandomForestRegressor y la validación cruzada
# -----------------------------------------------------------------------------
rf = RandomForestRegressor(random_state=42)

cv = KFold(n_splits=5, shuffle=True, random_state=42)  # 5-fold cross-validation

grid_search = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    scoring='neg_mean_squared_error',  # minimizamos MSE
    cv=cv,
    n_jobs=-1,
    verbose=1
)

# -----------------------------------------------------------------------------
# 8. Ajustar el modelo con búsqueda de hiperparámetros
# -----------------------------------------------------------------------------
grid_search.fit(X_train_scaled, y_train)

print("\n--- Mejor combinación de hiperparámetros ---")
print(grid_search.best_params_)

# Obtenemos el mejor modelo
best_rf = grid_search.best_estimator_

# -----------------------------------------------------------------------------
# 9. Predicción y evaluación en el conjunto de prueba
# -----------------------------------------------------------------------------
y_pred = best_rf.predict(X_test_scaled)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\n--- Resultados del Random Forest Óptimo ---")
print(f"MSE  = {mse:.2f}")
print(f"RMSE = {rmse:.2f}")
print(f"R^2  = {r2:.3f}")

# -----------------------------------------------------------------------------
# 10. Visualización de resultados: y_test vs. y_pred
# -----------------------------------------------------------------------------
plt.figure(figsize=(6, 6))
plt.scatter(y_test, y_pred, alpha=0.7, label="Predicciones")
plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    color='red', linestyle='--', label='Línea y=x'
)
plt.title("Random Forest: Valores Reales vs. Predicciones (Optimizado)")
plt.xlabel("Valor real (y_test)")
plt.ylabel("Predicción (y_pred)")
plt.legend()
plt.show()

# -----------------------------------------------------------------------------
# 11. Visualización de la importancia de características
# -----------------------------------------------------------------------------
importances = best_rf.feature_importances_
feature_names = ['infra_6_mes', 'contorno_avg', 'reborde_avg']

# Ordenamos de mayor a menor importancia
sorted_idx = np.argsort(importances)[::-1]
sorted_feature_names = [feature_names[i] for i in sorted_idx]
sorted_importances = importances[sorted_idx]

plt.figure(figsize=(6,4))
sns.barplot(x=sorted_importances, y=sorted_feature_names, color='skyblue')
plt.title("Importancia de características - Random Forest (Optimizado)")
plt.xlabel("Importancia")
plt.ylabel("Variable")
plt.show()
