# Análisis de Clustering con K-Means y PCA

Este proyecto utiliza dos técnicas fundamentales en el análisis de datos: **K-Means** (un algoritmo de clustering) y **PCA** (Análisis de Componentes Principales). El objetivo es agrupar datos en clusters y reducir su dimensionalidad para facilitar la visualización y la interpretación.

---

## Contenido

1. [Descripción del Proyecto](#descripción-del-proyecto)
2. [Técnicas Utilizadas](#técnicas-utilizadas)
   - [K-Means](#k-means)
   - [PCA](#pca)
3. [Implementación](#implementación)
   - [Preprocesamiento de Datos](#preprocesamiento-de-datos)
   - [Aplicación de K-Means](#aplicación-de-k-means)
   - [Aplicación de PCA](#aplicación-de-pca)
   - [Visualización de Resultados](#visualización-de-resultados)
4. [Comparación entre K-Means y PCA](#comparación-entre-k-means-y-pca)
5. [Interpretación de Resultados](#interpretación-de-resultados)
6. [Requisitos](#requisitos)
7. [Ejecución del Proyecto](#ejecución-del-proyecto)
8. [Contribuciones](#contribuciones)
9. [Licencia](#licencia)

---

## Descripción del Proyecto

Este proyecto tiene como objetivo:
- Aplicar el algoritmo **K-Means** para agrupar datos en clusters basados en su similitud.
- Utilizar **PCA** para reducir la dimensionalidad de los datos y visualizar los clusters en un espacio 2D.
- Comparar e interpretar los resultados de ambas técnicas para entender la estructura de los datos.

---

## Técnicas Utilizadas

### K-Means
- **Propósito**: Agrupamiento no supervisado.
- **Salida**: Etiquetas de cluster y centroides.
- **Uso**: Segmentación de datos, agrupamiento de clientes, etc.

### PCA
- **Propósito**: Reducción de dimensionalidad.
- **Salida**: Componentes principales que capturan la mayor varianza en los datos.
- **Uso**: Visualización de datos, reducción de ruido, preprocesamiento.

---

## Implementación

### Preprocesamiento de Datos
1. Cargar el conjunto de datos.
2. Eliminar valores faltantes y duplicados.
3. Codificar variables categóricas (si las hay).
4. Normalizar o estandarizar los datos (opcional).

### Aplicación de K-Means
1. Configurar el modelo K-Means con parámetros como `n_clusters`, `init`, y `random_state`.
2. Ajustar el modelo a los datos.
3. Asignar las etiquetas de cluster a cada punto de datos.

```python
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=5, random_state=111)
kmeans.fit(X)
data['Cluster'] = kmeans.labels_
```

### Aplicación de PCA
1. Configurar PCA con el número deseado de componentes (por ejemplo, 2 para visualización).
2. Transformar los datos al espacio de los componentes principales.

```python
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
pca_data = pca.fit_transform(X)
data['PC1'] = pca_data[:, 0]
data['PC2'] = pca_data[:, 1]
```

### Visualización de Resultados
1. Graficar los clusters de K-Means en el espacio de PCA.
2. Visualizar la varianza explicada por cada componente principal.

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Gráfico de clusters en PCA
plt.figure(figsize=(10, 6))
sns.scatterplot(x='PC1', y='PC2', hue='Cluster', data=data, palette='viridis', s=100, alpha=0.8)
plt.title('Clusters de K-Means en el espacio de PCA', fontsize=16)
plt.xlabel('Primer Componente Principal (PC1)', fontsize=12)
plt.ylabel('Segundo Componente Principal (PC2)', fontsize=12)
plt.legend(title='Cluster', fontsize=10)
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()
```

---

## Comparación entre K-Means y PCA

| Característica          | K-Means                          | PCA                              |
|-------------------------|----------------------------------|----------------------------------|
| **Propósito**           | Agrupamiento (clustering)        | Reducción de dimensionalidad     |
| **Salida**              | Etiquetas de cluster y centroides| Componentes principales          |
| **Uso**                 | Segmentación de datos            | Visualización y preprocesamiento |
| **Interpretación**      | Grupos basados en similitud      | Varianza explicada en los datos  |

---

## Interpretación de Resultados

- **K-Means**:
  - Los clusters representan grupos de puntos similares en el espacio de las características originales.
  - Los centroides indican el centro de cada cluster.

- **PCA**:
  - Los componentes principales capturan la mayor varianza en los datos.
  - La visualización en 2D o 3D ayuda a entender la estructura de los datos.

- **Relación**:
  - Si los clusters de K-Means están bien separados en el espacio de PCA, significa que K-Means ha capturado una estructura significativa.
  - Si los clusters se superponen, puede indicar que K-Means no ha encontrado una separación clara o que PCA no ha capturado suficiente varianza.

---

## Requisitos

- Python 3.x
- Bibliotecas:
  - `pandas`
  - `numpy`
  - `scikit-learn`
  - `matplotlib`
  - `seaborn`

Instala las dependencias con:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

---

## Ejecución del Proyecto

1. Clona el repositorio:
   ```bash
   git clone https://github.com/tu-usuario/tu-repositorio.git
   cd tu-repositorio
   ```
2. Ejecuta el script principal:
   ```bash
   python main.py
   ```

---

## Contribuciones

¡Las contribuciones son bienvenidas! Si deseas mejorar este proyecto, sigue estos pasos:
1. Haz un fork del repositorio.
2. Crea una rama con tu nueva funcionalidad (`git checkout -b nueva-funcionalidad`).
3. Realiza tus cambios y haz commit (`git commit -m 'Añadir nueva funcionalidad'`).
4. Haz push a la rama (`git push origin nueva-funcionalidad`).
5. Abre un Pull Request.

---

## Licencia

Este proyecto está bajo la licencia MIT. Para más detalles, consulta el archivo [LICENSE](LICENSE).

---

¡Gracias por usar este proyecto! Si tienes preguntas o sugerencias, no dudes en contactarme. 😊
