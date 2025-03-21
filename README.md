# Team Challenge: Pipelines en Machine Learning --- Team-GPTlines ����

## Descripción del Proyecto
Este repositorio contiene el desarrollo del Team Challenge orientado a la creación de pipelines utilizando Scikit-learn. El objetivo principal es construir un flujo de trabajo que incluya:
- **Preprocesamiento de datos:** Lectura, exploración (EDA), transformación (aplicación de log, escalado) y codificación de variables categóricas.
- **Tratamiento de desbalanceo:** Uso de SMOTE para balancear la variable objetivo (`Status`).
- **Modelado y Evaluación:** Entrenamiento y evaluación de modelos base (Regresión Logística, Random Forest y XGBoost) tanto con los datos originales como tras aplicar SMOTE.

## Descripción del Dataset
El dataset (`Application_Data.csv`) contiene información sobre solicitantes, con columnas como:
- **Applicant_ID:** Identificador único.
- **Applicant_Gender:** Género del solicitante (M/F).
- **Owned_Car, Owned_Realty, Owned_Mobile_Phone, Owned_Work_Phone, Owned_Phone, Owned_Email:** Indicadores binarios sobre bienes o medios de contacto.
- **Total_Children:** Número de hijos.
- **Total_Income:** Ingreso anual total.
- **Income_Type:** Tipo de fuente de ingreso (e.g., "Working", "Commercial associate", etc.).
- **Education_Type:** Nivel educativo (e.g., "Higher education", "Secondary education").
- **Family_Status:** Estado civil (e.g., "Married", "Single").
- **Housing_Type:** Tipo de vivienda.
- **Job_Title:** Cargo o profesión.
- **Total_Family_Members, Applicant_Age, Years_of_Working:** Datos demográficos y de experiencia.
- **Total_Bad_Debt, Total_Good_Debt:** Información sobre deudas.
- **Status:** Variable objetivo (1 para aprobado, 0 para rechazado).

> **Nota:** Se observa un fuerte desbalance en la variable `Status` (aproximadamente 99.5% de aprobados), lo cual motiva la aplicación de técnicas de sobremuestreo (SMOTE).

## Estructura del Repositorio

src 
    └── data 
        ├── Application_Data.csv \
      ├── Team-GPTlines_Pipelines_v0.ipynb \
      ├── Team-GPTlines_Pipelines_v1.ipynb \
      └── Team-GPTlines_Pipelines_v2.ipynb \
      └── Team-GPTlines_Pipelines_v3.ipynb # Versión final del notebook \
      └── Team-GPTlines__Pipelines_I.ipynb.ipynb \
      └── Team-GPTlines__Pipelines_II.ipynb.ipynb\ 

bootcampviztools.py 

.gitignore 

LICENSE README.md

- **Application_Data.csv**: Dataset principal con toda la información de los solicitantes.  
- **Team-GPTlines_Pipelines_v0.ipynb** / **v1.ipynb** / **v2.ipynb**: Notebooks de pruebas o versiones previas.  
- **Team-GPTlines_Pipelines_v3.ipynb**: Versión final con el flujo de trabajo definitivo:
  - Carga y exploración de datos (EDA)  
  - Preprocesamiento (transformaciones, escalado, codificación)  
  - División en train/test  
  - Aplicación de SMOTE  
  - Entrenamiento y evaluación de modelos (con y sin SMOTE)  
- **Team-GPTlines__Pipelines_I.ipynb.ipynb**
- **Team-GPTlines__Pipelines_II.ipynb.ipynb**
- **bootcampviztools.py**: Funciones de visualización auxiliares.  

## Dependencias y Entorno
El proyecto se ha desarrollado utilizando **Python (versión 3.12.7)**. Las principales librerías empleadas son:
- **Numpy** y **Pandas** para el manejo y transformación de datos.
- **Matplotlib** y **Seaborn** para la visualización.
- **Scikit-learn** para la construcción de modelos y preprocesamiento (incluyendo `StandardScaler`, `OneHotEncoder` y `train_test_split`).
- **XGBoost** para el modelo XGBClassifier.
- **imbalanced-learn** para la técnica de sobremuestreo con SMOTE.
- **bootcampviztools** para algunas visualizaciones específicas.


Ejecución del Proyecto
Exploración y Preprocesamiento:

Abrir el notebook Team-GPTlines_Pipelines_v3.ipynb en la carpeta src/data.
Ejecutar las celdas en orden para realizar la carga del dataset, análisis exploratorio (EDA), transformaciones (aplicación de log a variables seleccionadas, escalado y codificación de variables categóricas).
División de Datos y Balanceo:

Se realiza la división de los datos en conjuntos de entrenamiento y prueba.
Se aplica SMOTE para balancear la variable Status en el conjunto de entrenamiento.
Entrenamiento y Evaluación de Modelos:

Se entrenan tres modelos base: Regresión Logística, Random Forest y XGBoost.
Se evalúan los modelos tanto con los datos originales como con los datos sobremuestreados, generando métricas (accuracy, precision, recall, f1-score) y visualizando las matrices de confusión.
Comparación de Resultados:

Se crea una tabla resumen con los resultados obtenidos para cada modelo, lo que permite comparar el desempeño en función de la aplicación de SMOTE.
Resultados y Análisis
Los notebooks generan salidas gráficas (histogramas, mapas de calor de correlación y matrices de confusión) y tablas de resultados que evidencian:

La eficacia del preprocesamiento y la transformación de las variables.
La mejora en el desempeño de los modelos tras aplicar SMOTE, en especial en la predicción de la clase minoritaria.
La comparación entre los distintos modelos (Regresión Logística, Random Forest y XGBoost), destacando el buen desempeño de XGBoost.
Integrantes del Equipo:

Joaquín Villar Maldonado

Johann Strauss

Marco Fuchs

Rafael Neda

Miguel Angel Silva
