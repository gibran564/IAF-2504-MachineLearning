<div align="center">

![TecNM](assets/encabezado.png)

# Unidad 2 — Aprendizaje Supervisado: Regresión

### Machine Learning y Deep Learning · IAF-2504

![Python](https://img.shields.io/badge/Python-3.13-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-189AB4?style=for-the-badge&logo=xgboost&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)

![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=for-the-badge&logo=python&logoColor=white)
![Seaborn](https://img.shields.io/badge/Seaborn-4C72B0?style=for-the-badge&logo=python&logoColor=white)
![Status](https://img.shields.io/badge/Status-Completada-10b981?style=for-the-badge)

</div>

---

## Información del Alumno

|  |  |
|---|---|
| **Alumno** | Christian Gibrán Espituñal Villanueva |
| **No. de Control** | 20041243 |
| **Docente** | Dr. José Gabriel Rodríguez Rivas |
| **Asignatura** | Machine Learning y Deep Learning (IAF-2504) |
| **Unidad** | Unidad 2 — Aprendizaje Supervisado: Regresión |
| **Instituto** | Instituto Tecnológico de Durango — TecNM |
| **Periodo** | Enero – Junio 2025 |

---

## Objetivo de la Unidad

Implementar, evaluar y comparar los principales modelos de **regresión supervisada** aplicados a un dataset real de vehículos, explorando desde modelos lineales hasta técnicas de ensamble avanzadas. Cada práctica aborda un algoritmo distinto, documentando el fundamento teórico, la implementación en Python y el análisis crítico de resultados.

---

## Prácticas

| # | Práctica | Algoritmo | Dataset | Mejor R² | Estado |
|:---:|---|---|:---:|:---:|:---:|
| 1 | [Regresión Lineal Simple](Practica_1_Regresion_Lineal_Simple_Autos.ipynb) | Linear Regression (1 variable) | `autos2.csv` | ~0.81 | Completada |
| 2 | [Regresión Lineal Múltiple](Practica_2_Regresion_Lineal_Multiple_Autos.ipynb) | Linear Regression (múltiples variables) | `autos2.csv` | ~0.85 | Completada |
| 3 | [Árboles de Regresión](Practica_3_Arboles_Regresion_Autos.ipynb) | Decision Tree Regressor | `autos2.csv` | ~0.92 | Completada |
| 4 | [Random Forest](Practica_4_Random_Forest_Regresion_Autos.ipynb) | Random Forest Regressor | `autos2.csv` | ~0.92 | Completada |
| 5 | [Máquinas de Soporte (SVR)](Practica_5_Maquinas_de_Soporte_SVR_Autos.ipynb) | Support Vector Regression | `autos2.csv` | ~0.84 | Completada |
| 6 | [XGBoost](Practica_6_XGBoost_Regresion_Autos.ipynb) | Extreme Gradient Boosting | `autos2.csv` | ~0.92 | Completada |

### Resumen de métricas

| Modelo | RMSE (USD) | R² Test | Requiere Escalado | Interpretabilidad |
|---|:---:|:---:|:---:|:---:|
| Regresión Lineal Múltiple | ~$4,100 | ~0.85 | No | Alta |
| Árbol de Decisión | ~$3,200 | ~0.92 | No | Media |
| Random Forest | ~$3,100 | ~0.92 | No | Baja |
| SVR (RBF, C=100) | ~$4,365 | ~0.84 | **Sí** | Baja |
| XGBoost (optimizado) | ~$3,040 | ~0.92 | No | Media |

---

## Estructura del Repositorio

```
unidad_2/
│
├── assets/
│   └── encabezado.png
│
├── datasets/
│   └── autos2.csv
│
├── Practica_1_Regresion_Lineal_Simple_Autos.ipynb
├── Practica_2_Regresion_Lineal_Multiple_Autos.ipynb
├── Practica_3_Arboles_Regresion_Autos.ipynb
├── Practica_4_Random_Forest_Regresion_Autos.ipynb
├── Practica_5_Maquinas_de_Soporte_SVR_Autos.ipynb
├── Practica_6_XGBoost_Regresion_Autos.ipynb
│
└── README.md
```

---

## Dataset

Todas las prácticas utilizan el archivo **`datasets/autos2.csv`**, derivado del *Automobile Dataset* (UCI Machine Learning Repository). Contiene especificaciones técnicas y precios de mercado de ~205 vehículos.

| Columna | Tipo | Descripción |
|---|---|---|
| `horsepower` | Numérica | Potencia del motor (cv) |
| `engine-size` | Numérica | Cilindrada (cm³) |
| `city-mpg` | Numérica | Consumo urbano (millas/galón) |
| `wheel-base` | Numérica | Distancia entre ejes (pulgadas) |
| `bore` | Numérica | Diámetro del cilindro (pulgadas) |
| `price` | Numérica | **Variable objetivo** — Precio (USD) |
| *(+26 columnas adicionales)* | Mixto | Características técnicas del vehículo |

---

## Instalación y Configuración

### Requisitos previos

- Python 3.10 o superior
- `pip` actualizado

### 1. Clonar el repositorio

```bash
git clone https://github.com/usuario/ML_DeepLearning_ITD.git
cd ML_DeepLearning_ITD/unidad_2
```

### 2. Crear entorno virtual (recomendado)

```bash
python -m venv .venv

# Linux / macOS
source .venv/bin/activate

# Windows
.venv\Scripts\activate
```

### 3. Instalar dependencias

```bash
pip install -r requirements.txt
```

O bien, instalar los paquetes directamente:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn xgboost torch statsmodels jupyter ipykernel
```

### 4. Verificar soporte CUDA (opcional)

```python
import torch
print(torch.cuda.is_available())   # True si hay GPU disponible
print(torch.cuda.get_device_name(0))
```

> **Nota:** PyTorch se usa exclusivamente para detectar y reportar el hardware CUDA en cada práctica. Los modelos de sklearn y XGBoost se ejecutan en CPU.

### 5. Lanzar Jupyter

```bash
jupyter notebook
# o con JupyterLab:
jupyter lab
```

---

## Dependencias (`requirements.txt`)

```
numpy>=1.26
pandas>=2.2
matplotlib>=3.8
seaborn>=0.13
scikit-learn>=1.4
xgboost>=2.0
torch>=2.2
statsmodels>=0.14
jupyter>=1.0
ipykernel>=6.0
```

---

## Bibliografía

| # | Referencia |
|:---:|---|
| 1 | Géron, A. (2023). *Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow* (3.ª ed.). O'Reilly Media. |
| 2 | Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning* (2.ª ed.). Springer. Disponible en: https://web.stanford.edu/~hastie/ElemStatLearn/ |
| 3 | Chen, T., & Guestrin, C. (2016). *XGBoost: A Scalable Tree Boosting System*. KDD '16. https://doi.org/10.1145/2939672.2939785 |
| 4 | Pedregosa, F., et al. (2011). *Scikit-learn: Machine Learning in Python*. Journal of Machine Learning Research, 12, 2825–2830. |
| 5 | Smola, A. J., & Schölkopf, B. (2004). *A tutorial on support vector regression*. Statistics and Computing, 14(3), 199–222. |
| 6 | Breiman, L. (2001). *Random Forests*. Machine Learning, 45(1), 5–32. |
| 7 | UCI Machine Learning Repository. *Automobile Dataset*. https://archive.ics.uci.edu/dataset/10/automobile |
| 8 | Scikit-learn Developers. *API Reference*. https://scikit-learn.org/stable/api/index.html |
| 9 | XGBoost Developers. *XGBoost Documentation*. https://xgboost.readthedocs.io |
| 10 | PyTorch Team. *PyTorch Documentation*. https://pytorch.org/docs |

---

## Convenciones del Código

Todas las prácticas siguen el mismo estilo y estructura:

- `DATASET_PATH` centralizado como constante en la celda de configuración.
- Detección de GPU mediante PyTorch en la Sección 2 de cada notebook.
- Paleta de colores unificada: `#005F9E` (azul), `#E87722` (naranja), `#3DAD6B` (verde), `#C0392B` (rojo).
- DPI de figuras: 130. Estilo de ejes: `spines.top=False`, `spines.right=False`.
- Funciones reutilizables: `evaluar()`, `panel_evaluacion()`, `graficar_evaluacion()`.
- Sin escalado antes del split para evitar *data leakage*.

---

<div align="center">

**Christian Gibrán Espituñal Villanueva** — No. de Control: `20041243`

Ingeniería en Sistemas Computacionales — Instituto Tecnológico de Durango

*Asignatura IAF-2504 — Docente: Dr. José Gabriel Rodríguez Rivas*

</div>