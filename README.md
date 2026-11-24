# MLP superficial con TensorFlow-Keras sobre datos biomédicos de UCI

![python](https://img.shields.io/badge/python-3.10--3.12-blue)
![license](https://img.shields.io/badge/license-MIT-green)

Taller de la especialización en Inteligencia Artificial (UAO). Dos experimentos reproducibles con un
**perceptrón multicapa superficial** (1 o 2 capas ocultas) en **TensorFlow-Keras**, evaluados con datos
que nunca se usaron para entrenar.

- **Clasificación:** [BEED – Bangalore EEG Epilepsy Dataset](https://archive.ics.uci.edu/dataset/1134/beed:+bangalore+eeg+epilepsy+dataset)
  (8.000 registros × 16 canales EEG, 4 clases: sano, convulsión generalizada, convulsión focal, eventos de convulsión).
- **Regresión:** [ACTG 175](https://archive.ics.uci.edu/dataset/890/aids+clinical+trials+group+study+175)
  (2.139 pacientes con VIH). Se predice el CD4 a las ~20 semanas (`cd420`) usando solo datos **basales**.

---

## 1. Instalación

1. Instala **Python 3.10, 3.11 o 3.12**. TensorFlow suele dar problemas con versiones más nuevas.
2. Abre una terminal en la carpeta del proyecto y crea un entorno virtual **nuevo**:
   ```powershell
   python -m venv .venv
   .venv\Scripts\activate          # Linux/Mac: source .venv/bin/activate
   pip install -e ".[dev]"
   ```
3. Comprueba que todo quedó bien:
   ```powershell
   pytest -q                       # debe terminar con "18 passed"
   ```

> En Windows, TensorFlow ≥ 2.11 no usa la GPU de forma nativa; el proyecto corre bien en CPU. Si ves errores de
> DLL o de NumPy, usa un entorno virtual limpio o WSL2.

## 2. Datos

- **ACTG 175:** se descarga solo la primera vez (con `ucimlrepo`) y se guarda en `data/raw/actg175.csv`.
- **BEED:** `ucimlrepo` **no** puede importarlo. Descarga `BEED_Data.csv` desde su
  [página en UCI](https://archive.ics.uci.edu/dataset/1134/beed:+bangalore+eeg+epilepsy+dataset) y haz **una** de dos cosas:
  - guárdalo como `data/raw/beed.csv` (se encuentra solo), o
  - pasa la ruta al ejecutar: `--data-path data/raw/BEED_Data.csv`.
- Los CSV no se suben a GitHub (están en `.gitignore`).

## 3. Cómo ejecutarlo

### Opción A: notebook (recomendado para ver todo el análisis)

1. Abre `notebooks/taller_mlp_uci.ipynb` en VS Code.
2. Elige como kernel el Python de `.venv` (arriba a la derecha, **Select Kernel**).
3. En la celda de configuración escribe la ruta de BEED:
   ```python
   BEED_PATH = r"D:\ruta\a\tu\proyecto\data\raw\BEED_Data.csv"
   ```
4. **Restart** y **Run All**. Tarda varios minutos (18 configuraciones × 5 semillas por problema).

### Opción B: línea de comandos

```powershell
mlp-uci eda                                              # análisis exploratorio de ACTG 175
mlp-uci classify --data-path data/raw/BEED_Data.csv      # clasificación (BEED)
mlp-uci regress                                          # regresión (ACTG 175)
```

Opciones útiles:

- `--out CARPETA`: dónde guardar los resultados (por defecto `reports/...`).
- `--n-seeds N`: número de semillas para reportar media ± desviación (por defecto 5).
- `--max-epochs N`, `--patience N`: épocas máximas y paciencia del early stopping.
- `--include-sensitive` (solo regresión): incluye `race`, `gender`, `homo`, `drugs` como predictores.
- Con `make` (Linux/Mac/WSL): `make test`, `make classify`, `make regress`, `make all`.

### Qué se guarda en `reports/<tarea>/`

- `metrics.json`: métricas finales, mejores hiperparámetros y baselines.
- `grid_search.csv`: las 18 configuraciones con su pérdida de validación.
- `architectures.png`: **diagramas de las 3 arquitecturas**.
- `grid_heatmap.png` y `optimizer_curves.png`: comparación arquitectura × optimizador.
- `learning_curves.png`, `confusion_matrix.png` (clasificación) o `pred_vs_true.png` (regresión), importancia de variables.
- `model.keras` y `scaler.joblib`: modelo entrenado y escalador.

---

## 4. Qué se hizo (resumen del método)

1. **Tres conjuntos disjuntos.** Entrenamiento, validación y prueba (70/15/15). BEED con partición estratificada; ACTG con
   partición aleatoria (una fila por paciente). Los hiperparámetros se eligen solo con validación; la prueba se evalúa **una vez**.
2. **Preprocesamiento sin fuga.** `StandardScaler` ajustado solo con entrenamiento. En regresión también se escala el objetivo y
   las métricas se reportan en unidades originales. Variables categóricas de ACTG (`trt`, `strat`) con one-hot.
3. **Modelo.** `Entrada → [Dense(ReLU) → Dropout] × (1 o 2) → salida` (softmax para clasificar, lineal para regresión).
   Early stopping sobre `val_loss` y reducción de la tasa de aprendizaje en meseta. El código **rechaza más de 2 capas ocultas**.
4. **Tres arquitecturas:** `A_1x16`, `B_1x128` y `C_2x64_32`.
5. **Tres optimizadores:** Adam, SGD con momentum 0,9 y RMSprop. Además dropout {0; 0,2}: 18 combinaciones por problema.
6. **Robustez.** La mejor configuración se reentrena con 5 inicializaciones distintas y se reporta media ± desviación. Las gráficas
   salen de la semilla 0, no de la mejor.
7. **Baselines honestos.** Clase mayoritaria, regresión logística y random forest (clasificación); media, Ridge, random forest y
   "sin cambio" `cd420 = cd40` (regresión).
8. **Sin información del futuro (ACTG).** Se excluyen `cid`, `time`, `offtrt`, `cd820` (resultados o medidos después del
   tratamiento), `pidnum` (identificador) y `treat` (redundante). Las variables sensibles (`race`, `gender`, `homo`, `drugs`) se
   excluyen por defecto. Un test comprueba que ninguna llega al modelo.
9. **Interpretación.** Importancia por permutación sobre el conjunto de prueba y análisis exploratorio de ACTG 175.

## 5. Resultados

**BEED (prueba, media ± std de 5 semillas)**

| Modelo | Exactitud | Macro-F1 | AUC (OvR) |
|---|---|---|---|
| MLP superficial (1×128, Adam) | 0,957 ± 0,003 | 0,957 ± 0,003 | 0,9965 |
| Random forest | 0,961 | 0,961 | 0,9972 |
| Regresión logística | 0,465 | 0,468 | 0,659 |
| Clase mayoritaria | 0,250 | 0,100 | 0,500 |

- El MLP es competitivo con el random forest, pero **no lo supera**.
- 16 neuronas subajustan; una segunda capa no mejora frente a una capa ancha.
- Adam y RMSprop convergen más rápido que SGD.
- El entrenamiento llegó al límite de 300 épocas sin activar el early stopping.

**ACTG 175 (CD4 a la semana 20, prueba)**

| Modelo | RMSE | MAE | R² |
|---|---|---|---|
| MLP superficial (media de 5 semillas) | 113,0 | 86,8 | 0,361 |
| Ridge | 111,6 | 84,6 | 0,377 |
| Random forest | 116,5 | 90,3 | 0,321 |
| Sin cambio (`cd420 = cd40`) | 124,4 | 92,8 | 0,226 |
| Predictor de la media | 141,6 | 112,8 | −0,003 |

- El MLP supera los baselines ingenuos, pero **no a Ridge**: la relación es casi lineal.
- El CD4 basal (`cd40`) domina la predicción (importancia ≈ 0,65 de R²).
- Cambio medio de CD4 por brazo: ZDV −17, ddI +27, ZDV+Zal +19, ZDV+ddI +54.
- Las diferencias entre las 18 configuraciones están dentro del ruido.

## 6. Estructura del repositorio

```
src/mlp_uci/
  data.py         carga de BEED y ACTG 175, validación de columnas, particiones
  eda.py          análisis exploratorio de ACTG 175
  models.py       build_shallow_mlp(), arquitecturas y optimizadores
  training.py     entrenamiento con early stopping y búsqueda en malla
  evaluate.py     métricas, importancia por permutación, figuras
  baselines.py    modelos de referencia (scikit-learn)
  experiments.py  run_classification() y run_regression()
  cli.py          comando `mlp-uci classify | regress | eda`
tests/            18 pruebas sin conexión (sin solapamiento de conjuntos, sin variables post-basales, ...)
notebooks/        notebook del taller con resultados e interpretación
data/             instrucciones para los datos (los CSV no se suben)
```

### Contenido de las carpetas `data/raw`, `reports` y `tests`

**`data/raw/`** (datos de entrada)

- `BEED_Data.csv` o `beed.csv`: EEG de BEED, 8.000 filas con las columnas `X1`…`X16` (canales) y `y` (clase 0–3). Lo descargas tú.
- `actg175.csv`: tabla de ACTG 175, 2.139 pacientes. Se crea sola la primera vez que corres la regresión.
- `.gitkeep`: archivo vacío que mantiene la carpeta en Git. Los CSV **no** se suben (`.gitignore`).

**`reports/`** (salidas, se crean al ejecutar; en el repositorio original está vacía)

- `beed_classification/`: resultados de clasificación.
  - `metrics.json`, `grid_search.csv`
  - `architectures.png`, `grid_heatmap.png`, `optimizer_curves.png`
  - `learning_curves.png`, `confusion_matrix.png`, `channel_importance.png`
  - `model.keras`, `scaler.joblib`
- `actg175_regression/`: lo mismo, con `pred_vs_true.png` y `feature_importance.png` en lugar de la matriz de confusión y la importancia por canal.
- `actg175_eda/`: `eda_target_distribution.png`, `eda_cd40_vs_cd420.png`, `eda_treatment_effect.png`, `eda_correlations.png` y `eda_summary.json`.
- Las figuras y los `.json` se pueden subir a GitHub para mostrar resultados. `model.keras` y `scaler.joblib` están ignorados por Git porque son archivos pesados.

**`tests/`** (18 pruebas sin conexión, con datos sintéticos que imitan el esquema real; ejecutar con `pytest -q`)

- `conftest.py`: genera los CSV sintéticos de BEED y ACTG 175 que usan las pruebas.
- `test_data.py` (7): esquema de BEED, rechazo de columnas faltantes, particiones disjuntas y estratificadas, exclusión de variables post-basales y sensibles, filas sin objetivo.
- `test_models.py` (7): 1 o 2 capas ocultas permitidas y 3 rechazadas, que los valores por defecto cumplan lo que pide el taller (3 arquitecturas, 3 optimizadores), que cada optimizador entrene y validación de argumentos.
- `test_experiments.py` (4): flujo completo de clasificación y de regresión con sus archivos, rechazo de una partición desconocida y análisis exploratorio (EDA).

## 7. Limitaciones

- **BEED no trae identificador de sujeto**, así que no se puede dividir por persona. Si hay filas correlacionadas de una misma
  grabación, una partición aleatoria puede sobrestimar el rendimiento. Además, la descripción de UCI es inconsistente
  (8.000 vs. 16.000 segmentos), por lo que conviene tomar con cautela las exactitudes tan altas.
- **ACTG 175 es un ensayo de 1996**: los datos son históricos y no dicen nada sobre terapias actuales. Contiene atributos
  sensibles, excluidos por defecto.
- Con 321 pacientes de validación, las diferencias pequeñas entre hiperparámetros no son concluyentes.
- Proyecto educativo. **No es un dispositivo médico y no debe usarse para diagnóstico.**

## 8. Datos, citas y licencia

- BEED: Najmusseher & Banu (2024). *Feature Engineering for Epileptic Seizure Classification Using SeqBoostNet.*
  UCI, DOI [10.24432/C5K33B](https://doi.org/10.24432/C5K33B). Licencia CC BY 4.0.
- ACTG 175: Hammer et al. (1996). *A trial comparing nucleoside monotherapy with combination therapy in HIV-infected adults
  with CD4 cell counts from 200 to 500 per cubic millimeter.* N. Engl. J. Med. 335:1081-1090.
  UCI, DOI [10.24432/C5ZG8F](https://doi.org/10.24432/C5ZG8F). Revisa los términos en su página de UCI.
- Los datos no se redistribuyen en este repositorio.
- Código bajo licencia **MIT**; los datasets conservan sus licencias originales.

**Autor:** Juan Diego Estupiñán Rincón · Especialización en Inteligencia Artificial, UAO.
