import numpy as np
import pandas as pd
import plotly.express as px

# Preprocesamiento, modelado y validación
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, train_test_split, KFold
from sklearn.metrics import mean_squared_error, r2_score

# Para visualización
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Import necesario para proyecciones 3D

# -----------------------------------------------------------------------------
# 1. Carga de datos
# -----------------------------------------------------------------------------
data = pd.read_csv('datos_mezclados.csv')

# -----------------------------------------------------------------------------
# 2. Exploración inicial de los datos
# -----------------------------------------------------------------------------
print("\n--- Muestras de los datos originales ---")
print(data.head())

print("\n--- Descripción estadística original ---")
print(data.describe())

# -----------------------------------------------------------------------------
# 4. Ignorar valores atípicos en la columna 'reborde_avg' usando IQR
#    (puedes aplicar el mismo método a otras columnas si lo deseas).
# -----------------------------------------------------------------------------
Q1 = data['reborde_avg'].quantile(0.25)
Q3 = data['reborde_avg'].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Filtramos:
data = data[(data['reborde_avg'] >= lower_bound) & (data['reborde_avg'] <= upper_bound)]

print(f"\n--- Filtrado de outliers en 'reborde_avg' ---")
print(f"Rango permitido: [{lower_bound:.2f}, {upper_bound:.2f}]")
print(f"Registros restantes tras filtrar: {len(data)}")

# -----------------------------------------------------------------------------
# 6. Visualizaciones exploratorias (opcional, con los datos ya filtrados)
# -----------------------------------------------------------------------------
# 6.1. Pairplot
sns.pairplot(data[['infra_6_mes', 'contorno_avg', 'reborde_avg', 'volumen']],
             diag_kind='kde')
plt.suptitle("Pairplot tras filtrar outliers de reborde_avg", y=1.02)
plt.show()

# 6.2. Correlación (heatmap)
plt.figure(figsize=(6, 4))
correlaciones = data[['infra_6_mes', 'contorno_avg', 'reborde_avg', 'volumen']].corr()
sns.heatmap(correlaciones, annot=True, cmap="YlGnBu")
plt.title("Mapa de calor de correlaciones (datos filtrados)")
plt.show()

# 6.3. Scatter 3D coloreado por 'volumen'
fig = px.scatter_3d(
    data_frame=data,
    x='infra_6_mes',
    y='contorno_avg',
    z='reborde_avg',
    color='volumen',             # Asigna color basado en la variable 'volumen'
    color_continuous_scale='Viridis',  # Paleta de colores
    title="Gráfico 3D interactivo (Plotly) - Volumen coloreado"
)

# Opcional: personalizar el tamaño de los puntos
fig.update_traces(marker=dict(size=5))

# Mostrar el plot en modo interactivo (si estás en un Jupyter Notebook)
fig.show()

# -----------------------------------------------------------------------------
# EXPORTAR el gráfico a un archivo HTML para verlo en el navegador
# -----------------------------------------------------------------------------
fig.write_html("mi_scatter_3d_interactivo.html", auto_open=True)

# -----------------------------------------------------------------------------
# 7. Variables X e y
# -----------------------------------------------------------------------------
X = data[['infra_6_mes', 'contorno_avg', 'reborde_avg']].values
y = data['volumen'].values

# -----------------------------------------------------------------------------
# 8. División en entrenamiento y prueba
# -----------------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# -----------------------------------------------------------------------------
# 9. Construcción del Pipeline (PolynomialFeatures + StandardScaler + Ridge)
# -----------------------------------------------------------------------------
pipeline = Pipeline([
    ("poly", PolynomialFeatures(include_bias=False)),  # grado se define en la búsqueda
    ("scaler", StandardScaler()),
    ("ridge", Ridge())  # alpha se define en la búsqueda
])

# -----------------------------------------------------------------------------
# 10. Definir la rejilla de hiperparámetros
# -----------------------------------------------------------------------------
param_grid = {
    "poly__degree": [1, 2, 3, 4],
    "ridge__alpha": [0.01, 0.1, 1, 10]
}

# -----------------------------------------------------------------------------
# 11. Configurar Cross-Validation con KFold
# -----------------------------------------------------------------------------
cv = KFold(n_splits=5, shuffle=True, random_state=42)

# -----------------------------------------------------------------------------
# 12. GridSearchCV
# -----------------------------------------------------------------------------
grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    scoring="neg_mean_squared_error",
    cv=cv,
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train, y_train)

# -----------------------------------------------------------------------------
# 13. Resultados de la búsqueda
# -----------------------------------------------------------------------------
print("\n--- Resultados de GridSearchCV ---")
print("Mejores hiperparámetros encontrados:")
print(grid_search.best_params_)

print("\nMejor puntuación de CV (MSE negativo):")
print(grid_search.best_score_)

# -----------------------------------------------------------------------------
# 14. Evaluación en el conjunto de prueba
# -----------------------------------------------------------------------------
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

mse_test = mean_squared_error(y_test, y_pred)
rmse_test = np.sqrt(mse_test)
r2_test = r2_score(y_test, y_pred)

print(f"\n--- Métricas en el test set ---")
print(f"MSE  = {mse_test:.3f}")
print(f"RMSE = {rmse_test:.3f}")
print(f"R^2  = {r2_test:.3f}")

# -----------------------------------------------------------------------------
# 15. Visualización de y_test vs. y_pred
# -----------------------------------------------------------------------------
plt.figure(figsize=(6, 6))
plt.scatter(y_test, y_pred, alpha=0.7, label="Predicciones")

# Línea diagonal (y = x)
plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    color='red',
    linestyle='--',
    label='Línea y = x'
)

# Definimos un margen del 10% alrededor de la línea y = x
margin_ratio = 0.1  # 10% de margen (ajusta según tu criterio)

# Calculamos límites superior e inferior para la banda
band_lower = y_test * (1.0 - margin_ratio)
band_upper = y_test * (1.0 + margin_ratio)

# Para poder hacer un fill_between, necesitamos los valores ordenados
sort_idx = np.argsort(y_test)  # indices que ordenan y_test de menor a mayor

# Rellenamos la banda
plt.fill_between(
    y_test[sort_idx],
    band_lower[sort_idx],
    band_upper[sort_idx],
    color='blue',
    alpha=0.1,
    label=f"Banda ±{int(margin_ratio*100)}%"
)

plt.xlabel("Valor real (y_test)")
plt.ylabel("Predicción (y_pred)")
plt.title("Comparación de valores reales vs. predicciones con banda de tolerancia")
plt.legend()
plt.show()
