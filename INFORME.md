# Informe: perceptrón multicapa superficial para clasificación de EEG y regresión de CD4

**Autor:** Juan Diego Estupiñán Rincón · Especialización en Inteligencia Artificial, Universidad Autónoma de Occidente
**Fecha:** Noviembre de 2026
**Código y reproducción:** ver [README.md](README.md) · **Análisis ejecutable:** [notebooks/taller_mlp_uci.ipynb](notebooks/taller_mlp_uci.ipynb)

---

## 1. Introducción

Las redes neuronales artificiales se usan cada vez más en el análisis de datos biomédicos, pero un resultado numérico solo
es útil si se obtuvo con una metodología que lo haga creíble. Este trabajo entrena **perceptrones multicapa (MLP) superficiales**,
es decir, con una o dos capas ocultas, en **TensorFlow-Keras** para resolver dos problemas con conjuntos de datos del
repositorio UCI:

1. **Clasificación** de señales de electroencefalografía (EEG) en cuatro estados: sano y tres tipos de actividad convulsiva
   (dataset BEED).
2. **Regresión** del conteo de linfocitos CD4 a las ~20 semanas en pacientes con VIH a partir de datos basales (dataset ACTG 175).

Los objetivos son:

- Entrenar y validar cada modelo con datos que no se usaron para entrenar.
- Comparar al menos tres optimizadores y tres arquitecturas, con un máximo de dos capas ocultas, y presentar sus diagramas.
- Comparar el MLP con modelos de referencia (*baselines*) para saber si la red realmente aporta.
- Documentar las decisiones que evitan resultados inflados (fuga de datos) y las limitaciones de cada conjunto de datos.

El trabajo es educativo y **no constituye un dispositivo médico**.

## 2. Marco teórico

### 2.1 Perceptrón multicapa

Un MLP es una red de neuronas organizadas en capas. Cada neurona de una capa densa calcula una combinación lineal de sus
entradas y le aplica una función de activación:

$$ h = \sigma(Wx + b) $$

donde $W$ son los pesos, $b$ el sesgo y $\sigma$ la activación. Una red **superficial** tiene una o dos capas ocultas; una
**profunda** tiene muchas. Por el teorema de aproximación universal, una sola capa oculta con suficientes neuronas puede aproximar
funciones continuas arbitrarias (Hornik et al., 1989). Para datos tabulares de tamaño moderado, como los de este trabajo, las
redes superficiales suelen ser suficientes y son más fáciles de entrenar e interpretar.

### 2.2 Activaciones, salida y función de pérdida

- **ReLU**, $\max(0, z)$, en las capas ocultas: barata de calcular y mitiga el desvanecimiento del gradiente.
- **Softmax** en la salida de clasificación, que convierte los valores en probabilidades de cada clase, con **entropía cruzada** como pérdida.
- **Salida lineal** en regresión, con **error cuadrático medio** como pérdida.

### 2.3 Entrenamiento y optimizadores

La red aprende minimizando la pérdida con descenso de gradiente, calculado mediante retropropagación (Rumelhart et al., 1986).
Se compararon tres optimizadores:

- **SGD con momentum:** actualiza los pesos en la dirección del gradiente acumulando una "velocidad" que suaviza el avance
  (momentum = 0,9).
- **RMSprop:** adapta la tasa de aprendizaje de cada parámetro dividiendo por una media móvil de los gradientes al cuadrado
  (Hinton, 2012).
- **Adam:** combina el momentum con la adaptación por parámetro de RMSprop (Kingma & Ba, 2015).

### 2.4 Regularización y control del sobreajuste

El **sobreajuste** ocurre cuando el modelo memoriza el entrenamiento y generaliza mal. Se controló con:

- **Dropout:** apagar aleatoriamente una fracción de neuronas durante el entrenamiento (Srivastava et al., 2014).
- **Parada temprana (*early stopping*):** detener el entrenamiento cuando la pérdida de validación deja de mejorar y restaurar
  los mejores pesos (Prechelt, 1998).
- **Reducción de la tasa de aprendizaje en meseta.**

### 2.5 Partición de datos y fuga de información

Se usan tres conjuntos disjuntos: **entrenamiento** (ajustar pesos), **validación** (elegir hiperparámetros y detener el
entrenamiento) y **prueba** (estimar el rendimiento final, evaluado una sola vez). Hay **fuga de datos** cuando información del
conjunto de prueba, o del futuro respecto al momento de la predicción, entra al entrenamiento. Ejemplos: ajustar el escalador con
todos los datos, usar como predictor una variable medida después del desenlace, o repetir al mismo paciente en entrenamiento y
prueba.

### 2.6 Métricas

- **Clasificación:** exactitud (*accuracy*), F1 macro (promedio del F1 de cada clase, sin sesgo por clase) y AUC-ROC uno contra
  el resto, junto con la matriz de confusión.
- **Regresión:** raíz del error cuadrático medio (RMSE), error absoluto medio (MAE) y coeficiente de determinación $R^2$, que
  compara el modelo con predecir siempre la media ($R^2 = 0$ equivale a la media; negativo es peor que la media).

### 2.7 Modelos de referencia e importancia de variables

Un resultado solo se interpreta frente a una referencia. Se usaron regresión logística y Ridge (modelos lineales), y *random
forest* (Breiman, 2001), un conjunto de árboles de decisión. La **importancia por permutación** mide cuánto empeora el
rendimiento al barajar una variable en el conjunto de prueba.

## 3. Descripción del problema

### 3.1 Clasificación de EEG (BEED)

BEED (*Bangalore EEG Epilepsy Dataset*, Najmusseher & Banu, 2024) contiene **8.000 registros** de 16 canales de EEG (`X1`…`X16`)
con cuatro clases balanceadas (2.000 por clase): **sano**, **convulsión generalizada**, **convulsión focal** y **eventos de
convulsión**. El problema es predecir el estado a partir de los 16 canales. La detección automática de actividad epiléptica
podría apoyar el análisis clínico del EEG, que es largo y requiere experiencia.

### 3.2 Regresión del CD4 (ACTG 175)

ACTG 175 proviene de un ensayo clínico aleatorizado (Hammer et al., 1996) con **2.139 adultos con VIH** y CD4 entre 200 y 500
células/mm³, repartidos en cuatro brazos de tratamiento: ZDV, ZDV+ddI, ZDV+Zal y ddI. El objetivo es predecir el CD4 a las
~20 semanas (`cd420`) usando solo información disponible **antes** del tratamiento. Anticipar la respuesta inmunológica ayuda a
pensar en cómo individualizar el seguimiento.

### 3.3 Decisión sobre el conjunto de regresión

Inicialmente se probó *Parkinsons Telemonitoring* (UCI 189), pero cada paciente aporta unas 200 grabaciones. Una partición
aleatoria pone al mismo paciente en entrenamiento y prueba e infla el rendimiento, y una partición por paciente daba un
rendimiento bajo. Se cambió a ACTG 175, donde **cada fila es un paciente distinto** y una partición aleatoria no filtra
información entre conjuntos.

## 4. Planteamiento de la solución

### 4.1 Preparación de los datos

- **Particiones:** 70 % entrenamiento, 15 % validación, 15 % prueba. BEED con partición **estratificada**
  (5.600 / 1.200 / 1.200 registros); ACTG 175 con partición aleatoria (1.497 / 321 / 321 pacientes).
- **Escalado:** `StandardScaler` ajustado solo con entrenamiento. En regresión también se escala el objetivo y las métricas se
  reportan en las unidades originales.
- **Variables de ACTG 175:** numéricas (`age`, `wtkg`, `karnof`, `preanti`, `cd40`, `cd80`), binarias y categóricas con
  *one-hot* (`trt`, `strat`).
- **Exclusiones para evitar fuga del futuro:** `cid`, `time`, `offtrt` y `cd820` (medidos después de iniciar el tratamiento o
  resultados), `pidnum` (identificador) y `treat` (redundante con `trt`). Las variables sensibles (`race`, `gender`, `homo`,
  `drugs`) se excluyen por defecto. Una prueba automática verifica que ninguna llegue al modelo.

### 4.2 Arquitecturas

Cada modelo es `Entrada → [Dense(ReLU) → Dropout] × (1 o 2) → salida`. El código lanza un error si se piden más de dos capas
ocultas. Los diagramas se generan en `reports/*/architectures.png`.

| Nombre | Capas ocultas | Descripción |
|---|---|---|
| `A_1x16` | 1 × 16 | Una capa pequeña |
| `B_1x128` | 1 × 128 | Una capa ancha |
| `C_2x64_32` | 64 → 32 | Dos capas ocultas |

### 4.3 Optimizadores y búsqueda de hiperparámetros

Se cruzaron las 3 arquitecturas con 3 optimizadores (Adam, SGD con momentum 0,9 y RMSprop) y dropout $\in \{0; 0{,}2\}$:
**18 configuraciones por problema**, seleccionadas por la menor pérdida de validación. Tasas de aprendizaje por defecto: 1e-3
(Adam y RMSprop) y 1e-2 (SGD). Entrenamiento con lotes de 128, máximo 300 épocas y paciencia de 20.

### 4.4 Evaluación

- La mejor configuración se **reentrena con 5 inicializaciones** distintas; se reporta media ± desviación en prueba.
- Las gráficas provienen de la semilla 0, no de la mejor, para no escoger a conveniencia.
- La prueba se evalúa una sola vez.
- **Baselines:** clase mayoritaria, regresión logística y random forest (clasificación); media, Ridge, random forest y "sin
  cambio" (`cd420 = cd40`) en regresión.

### 4.5 Implementación

Paquete de Python (`src/mlp_uci`) con 18 pruebas automáticas sin conexión, interfaz de línea de comandos (`mlp-uci`), análisis
exploratorio y notebook con las interpretaciones. Herramientas: TensorFlow-Keras, scikit-learn, pandas, matplotlib y ruff.

## 5. Resultados

Las figuras mencionadas se generan en `reports/` al ejecutar el notebook (ver README).

### 5.1 Clasificación de EEG (BEED)

La mejor configuración fue `B_1x128` con Adam, sin dropout (2.692 parámetros).

| Modelo | Exactitud | F1 macro | AUC (OvR) |
|---|---|---|---|
| **MLP superficial** (media de 5 semillas) | 0,957 ± 0,003 | 0,957 ± 0,003 | 0,9965 |
| Random forest | 0,961 | 0,961 | 0,9972 |
| Regresión logística | 0,465 | 0,468 | 0,659 |
| Clase mayoritaria | 0,250 | 0,100 | 0,500 |

**Arquitecturas y optimizadores** (`grid_heatmap.png`, pérdida de validación):

| Arquitectura | Adam | RMSprop | SGD |
|---|---|---|---|
| `A_1x16` | 0,277 | 0,265 | 0,276 |
| `B_1x128` | 0,112 | 0,112 | 0,185 |
| `C_2x64_32` | 0,113 | 0,118 | 0,119 |

**Observaciones:**

- La red supera ampliamente a los modelos lineales: el problema es no lineal. Con 16 neuronas el modelo subajusta.
- `B_1x128` y `C_2x64_32` rinden igual: una segunda capa no aporta frente a una capa ancha.
- Adam y RMSprop convergen igual de rápido; SGD es más lento y solo se acerca con la arquitectura de dos capas.
- El random forest queda 0,4 puntos por encima; la diferencia es comparable a la variación entre semillas, por lo que el MLP es
  **competitivo, no superior**.
- La pérdida de validación seguía bajando a las 300 épocas, así que la parada temprana no llegó a activarse; el límite de
  épocas, y no el sobreajuste, detuvo el entrenamiento. La brecha entre entrenamiento (≈ 0,99) y validación (≈ 0,96) indica
  sobreajuste leve.
- **Matriz de confusión** (semilla 0): la clase sana se separa casi perfecto (99 % de recall). La confusión se concentra
  entre convulsión focal y eventos de convulsión (7 % y 5 %), las más parecidas.
- **Importancia por canal:** muy plana (la caída al barajar cada canal está entre 0,11 y 0,17); la información está repartida
  entre los 16 canales, y `X13` y `X14` aportan menos.

### 5.2 Regresión del CD4 (ACTG 175)

La mejor configuración por validación fue `B_1x128` con SGD y dropout 0,2 (2.689 parámetros).

| Modelo | RMSE | MAE | R² |
|---|---|---|---|
| **MLP superficial** (media de 5 semillas) | 113,0 | 86,8 | 0,361 |
| Ridge | 111,6 | 84,6 | 0,377 |
| Random forest | 116,5 | 90,3 | 0,321 |
| Sin cambio (`cd420 = cd40`) | 124,4 | 92,8 | 0,226 |
| Predictor de la media | 141,6 | 112,8 | −0,003 |

**Análisis exploratorio.** No hay valores faltantes y la correlación entre `cd40` y `cd420` es 0,584. El cambio medio de CD4
por brazo de tratamiento (IC 95 %) fue:

| Brazo | n | Cambio medio | IC 95 % |
|---|---|---|---|
| ZDV | 532 | −17,1 | [−26,0; −8,2] |
| ZDV+ddI | 522 | +54,4 | [+42,1; +66,8] |
| ZDV+Zal | 524 | +19,3 | [+9,6; +28,9] |
| ddI | 561 | +26,9 | [+17,4; +36,3] |

El intervalo de ZDV+ddI no se solapa con el de ZDV solo. Es una asociación descriptiva en un ensayo aleatorizado, no un análisis
causal formal.

**Observaciones:**

- El MLP supera claramente a la media y al baseline clínico "sin cambio", pero **no supera a Ridge**: la relación es casi
  lineal y la red no aporta ventaja.
- `cd40` domina la predicción: al barajarla, el $R^2$ cae ≈ 0,65. Le siguen el tratamiento (`trt_0`, ≈ 0,05) y variables con
  efectos pequeños.
- Los puntos predichos se comprimen hacia la media (se subestiman los CD4 altos); los residuos son aproximadamente simétricos.
- Las 18 configuraciones difieren menos de 0,05 en pérdida de validación: la elegida no es significativamente mejor que las
  demás. Con 321 pacientes de validación esa diferencia es ruido.

## 6. Conclusiones

1. Un MLP superficial resuelve bien la clasificación de EEG (95,7 % de exactitud), con una capa de 128 neuronas y Adam. Sin
   embargo, un random forest sin ajuste fino lo iguala (96,1 %), por lo que la red no es necesariamente la mejor herramienta.
2. En la regresión del CD4 el MLP (R² = 0,36) supera a los modelos ingenuos, pero no a Ridge: **más complejidad no implica
   mejor resultado** cuando la relación es casi lineal.
3. Adam y RMSprop convergen más rápido que SGD, y dos capas no mejoran frente a una capa ancha, lo que respalda el uso de redes
   superficiales en datos tabulares de este tamaño.
4. La metodología fue tan importante como la arquitectura: particiones sin fuga, escalado solo con entrenamiento, evaluación
   única en prueba, varias semillas y baselines honestos. El caso de Parkinsons motivó la elección del conjunto de regresión.
5. Aprendizaje principal: reportar con honestidad que el modelo avanzado no gana es parte de un resultado defendible.

**Limitaciones.**

- BEED no trae identificador de sujeto, por lo que no se pudo dividir por persona; si hay filas correlacionadas de una misma
  grabación, la exactitud podría estar sobrestimada. La descripción de UCI también es inconsistente (8.000 frente a 16.000
  segmentos), por lo que las exactitudes tan altas se toman con cautela.
- ACTG 175 es un ensayo de 1996 y no refleja las terapias actuales; contiene atributos sensibles (excluidos por defecto).
- Con 321 pacientes de validación, las diferencias pequeñas entre hiperparámetros no son concluyentes.

**Trabajo futuro.** Probar una capa de 256 neuronas y más épocas en BEED, validación cruzada, y medir el efecto de incluir las
variables sensibles en ACTG (`--include-sensitive`).

## 7. Referencias

- Abadi, M., et al. (2016). TensorFlow: A system for large-scale machine learning. *Proc. 12th USENIX Symposium on Operating
  Systems Design and Implementation (OSDI)*, 265-283.
- Breiman, L. (2001). Random forests. *Machine Learning, 45*(1), 5-32.
- Chollet, F., et al. (2015). *Keras*. https://keras.io
- Hammer, S. M., Katzenstein, D. A., Hughes, M. D., et al. (1996). A trial comparing nucleoside monotherapy with combination
  therapy in HIV-infected adults with CD4 cell counts from 200 to 500 per cubic millimeter. *New England Journal of
  Medicine, 335*(15), 1081-1090. Datos: UCI Machine Learning Repository, https://doi.org/10.24432/C5ZG8F
- Hinton, G. (2012). Neural networks for machine learning, lecture 6e: RMSprop. Coursera.
- Hornik, K., Stinchcombe, M., & White, H. (1989). Multilayer feedforward networks are universal approximators. *Neural
  Networks, 2*(5), 359-366.
- Kingma, D. P., & Ba, J. (2015). Adam: A method for stochastic optimization. *International Conference on Learning
  Representations (ICLR)*.
- Najmusseher, & Banu, S. (2024). Feature engineering for epileptic seizure classification using SeqBoostNet. *International
  Journal of Computing and Digital Systems*. Datos: UCI Machine Learning Repository, https://doi.org/10.24432/C5K33B
- Pedregosa, F., et al. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research, 12*, 2825-2830.
- Prechelt, L. (1998). Early stopping - but when? En G. B. Orr & K.-R. Müller (Eds.), *Neural Networks: Tricks of the Trade*
  (pp. 55-69). Springer.
- Rumelhart, D. E., Hinton, G. E., & Williams, R. J. (1986). Learning representations by back-propagating errors. *Nature,
  323*, 533-536.
- Srivastava, N., Hinton, G., Krizhevsky, A., Sutskever, I., & Salakhutdinov, R. (2014). Dropout: A simple way to prevent neural
  networks from overfitting. *Journal of Machine Learning Research, 15*, 1929-1958.
