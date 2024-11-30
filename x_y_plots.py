import pandas as pd
import matplotlib.pyplot as plt

# Load your data
data = pd.read_csv('breast_implant_data_1.csv')

# Define X and y variables
X = data[['volumen_pre', 'proyeccion', 'reborde', 'inframamaria']]
y = data[['base_implante_colocado', 'volumen_colocado', 'proyeccion_medida']]

# Create the plot
fig, axes = plt.subplots(len(X.columns), len(y.columns), figsize=(15, 10))

# Loop over each combination of X and y variables
for i, x_col in enumerate(X.columns):
    for j, y_col in enumerate(y.columns):
        ax = axes[i, j]
        ax.scatter(X[x_col], y[y_col], alpha=0.5)
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        ax.set_title(f'{x_col} vs {y_col}')

plt.tight_layout()
plt.show()
