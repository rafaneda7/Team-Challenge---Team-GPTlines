# Team Challenge: Pipelines en Machine Learning

## Descripción del Proyecto
Este repositorio contiene el desarrollo del Team Challenge orientado a la creación de pipelines utilizando **Scikit-learn**. El objetivo principal es construir un flujo de trabajo integral que incluya:

- **Preprocesamiento de datos**  
  (lectura, análisis exploratorio, transformaciones como log y escalado, codificación de variables categóricas, etc.).
- **Tratamiento de desbalanceo**  
  (uso de SMOTE para equilibrar la variable objetivo `Status`).
- **Modelado y Evaluación**  
  (entrenamiento y comparación de distintos modelos: Regresión Logística, Random Forest y XGBoost, con y sin balanceo).
- **Optimización de Hiperparámetros**  
  (uso de `GridSearchCV` y validación cruzada `StratifiedKFold`).

El objetivo de negocio es desarrollar un **modelo predictivo** que ayude a determinar la aprobación (1) o rechazo (0) de solicitudes, optimizando la toma de decisiones en una posible entidad financiera.

---

## Explicación del Dataset
El dataset principal (`Application_Data.csv`) contiene información detallada sobre cada solicitante. Algunas columnas destacadas son:

- **Applicant_ID:** Identificador único de cada solicitante.  
- **Applicant_Gender:** Género (M/F).  
- **Owned_Car, Owned_Realty, Owned_Mobile_Phone, etc.:** Indicadores binarios de posesión de bienes o medios de contacto.  
- **Total_Children, Total_Income, Income_Type:** Información socioeconómica.  
- **Education_Type, Family_Status, Housing_Type:** Datos sobre educación, estado civil y tipo de vivienda.  
- **Job_Title:** Cargo o profesión.  
- **Applicant_Age, Years_of_Working:** Datos demográficos y experiencia laboral.  
- **Total_Bad_Debt, Total_Good_Debt:** Deudas malas o buenas.  
- **Status:** Variable objetivo (1 = aprobado, 0 = rechazado).

> **Nota:** El dataset está **desbalanceado** (cerca del 99.5% de aprobados), por lo que se emplean técnicas de sobremuestreo (SMOTE) para mejorar la capacidad de predicción en la clase minoritaria.

---

## Estructura del Repositorio

ssrc 
  ├── data │ 
    ├── Application_Data.csv │ 
    ├── Application_Data_train.csv │ 
    ├── Application_Data_test.csv │ 
    ├── Application_Data_train2.csv │ 
    └── Application_Data_test2.csv 
  ├── notebooks │ 
    ├── Team-GPTlines_Pipelines_v0.ipynb │ 
    ├── Team-GPTlines_Pipelines_v1.ipynb │
    ├── Team-GPTlines_Pipelines_v2.ipynb │
    ├── Team-GPTlines_Pipelines_v3.ipynb │ 
    └── bootcampviztools.py 
  └── result_notebooks 
    ├── Team-GPTlines_Pipelines_I.ipynb 
    ├── Team-GPTlines_Pipelines_II.ipynb 
    ├── modelo_pipeline.joblib 
    └── modelo_pipeline.pkl
.gitignore
LICENSE 
README.md

---

- **Application_Data.csv**: El dataset, denominado Application_Data.csv, contiene información detallada sobre solicitantes de crédito o productos financieros. Cada registro representa a un individuo y reúne tanto datos demográficos como económicos y de comportamiento crediticio.   
- **Team-GPTlines_Pipelines_v0.ipynb / v1.ipynb / v2.ipynb**: Notebooks de pruebas o versiones previas.  
- **Team-GPTlines_Pipelines_v3.ipynb**: Versión final con el flujo de trabajo definitivo:
  - Carga y exploración de datos (EDA).  
  - Preprocesamiento (transformaciones, escalado, codificación).  
  - División en train/test.  
  - Aplicación de SMOTE.  
  - Entrenamiento y evaluación de modelos (con y sin SMOTE), incluyendo optimización de hiperparámetros.  
- **bootcampviztools.py**: Funciones de visualización auxiliares.  

---

## Dependencias y Entorno
El proyecto se ha desarrollado utilizando **Python (versión 3.12.7)**. Las principales librerías empleadas son:

- **Numpy** y **Pandas** para el manejo y transformación de datos.  
- **Matplotlib** y **Seaborn** para la visualización.  
- **Scikit-learn** para la construcción de modelos y preprocesamiento (incluyendo `StandardScaler`, `OneHotEncoder` y `train_test_split`).  
- **XGBoost** para el modelo `XGBClassifier`.  
- **imbalanced-learn** para la técnica de sobremuestreo con SMOTE.  
- **joblib** y **pickle** para guardar y cargar el pipeline entrenado.  
- **bootcampviztools** para algunas visualizaciones específicas.


Ejecución del Proyecto
Exploración y Preprocesamiento

Abrir el notebook Team-GPTlines_Pipelines_v3.ipynb (versión final) o el que corresponda en la carpeta src/notebooks.
Ejecutar las celdas en orden para realizar la carga del dataset, análisis exploratorio (EDA), transformaciones (aplicación de log, escalado, codificación de variables categóricas).
División de Datos y Balanceo

Se realiza la división de los datos en conjuntos de entrenamiento y prueba.
Se aplica SMOTE para balancear la variable Status en el conjunto de entrenamiento.
Entrenamiento y Evaluación de Modelos

Se entrenan tres modelos base: Regresión Logística, Random Forest y XGBoost.
Se evalúan los modelos tanto con los datos originales como con los datos sobremuestreados, generando métricas (accuracy, precision, recall, f1-score) y visualizando las matrices de confusión.
Optimización de Hiperparámetros

Se utilizan GridSearchCV y StratifiedKFold para encontrar la mejor configuración de hiperparámetros.
Se compara el desempeño de cada modelo según las métricas definidas.
Comparación de Resultados

Se crea una tabla resumen con los resultados obtenidos para cada modelo, lo que permite comparar el desempeño en función de la aplicación de SMOTE.
Se elige el modelo final y se guardan los pipelines entrenados (en modelo_pipeline.pkl o modelo_pipeline.joblib).
Resultados y Análisis
Los notebooks generan salidas gráficas (histogramas, mapas de calor de correlación y matrices de confusión) y tablas de resultados que evidencian:

La eficacia del preprocesamiento y la transformación de las variables.
La mejora en el desempeño de los modelos tras aplicar SMOTE, en especial en la predicción de la clase minoritaria.
La comparación entre los distintos modelos (Regresión Logística, Random Forest y XGBoost), destacando el buen desempeño de XGBoost.

Integrantes del Equipo:

- Joaquín Villar Maldonado
- Johann Strauss
- Marco Fuchs
- Rafael Neda
- Miguel Angel Silva