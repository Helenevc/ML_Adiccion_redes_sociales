🌐 [English](#readme-en) | [Español](##readme-es)

## README (EN)

**ML_Social_Networking_addictions**
-----------

ML_Social_Network_Addictions is a Machine Learning analysis developed as part of an individual assignment in the Data Science bootcamp at The Bridge, in July 2025.


**🧭 Table of Contents**
-----------

- Objective
- Dataset used
- Proposed Solution
- Useful libraries
- Repository content

  

---

**🎯Objective**
------------

Welcome to my project in which we want to analyse the data and look for the best model to predict the level of addiction of social network on student users. 


**📂Dataset used**
----------------------------

To make this model we use a Kaggle (url: https://www.kaggle.com/code/adilshamim8/social-media-addiction-among-students/notebook) dataset on social network addictions of students, which has 13 columns and 705 instances. 

This CSV file contains survey responses from 16-25 year old students in several countries, collecting their social media usage patterns along with key personal, academic and well-being indicators. 

It includes numerical and categorical fields which allows the exploratory analysis and predictive modelling of the relationships between social network addiction and various life outcomes.

**🧠Proposed Solution**
----------

The work started with the analysis of the dataset and the available features. 
We are facing an unbalanced, multi-class classification problem.
After understanding the dataset, a mini EDA of the variables (one by one and features togheter) was performed. 
Then several features selection methodologies were used in order to make a more efficient model, with various types of visualisations from Seaborn and Matplotlib.

Several models (8) were compared and we finally have decided to optimise the one that best met our needs (Random Forest Classifier).
We´ve predicted our test sets against the best model, with its best selection of features, and after its best optimization we have evaluated its result with a confusion matrix. The results seemed satisfactory for the problem we had so we saved the model, and you can find it here (“src/models”).


**📦Useful libraries**
----------
| Library                        | Principal Use                                                            |
|---------------------------------|---------------------------------------------------------------------------|
| **numpy**                       | Numerical calculations and operations with arrays                              |
| **pandas**                      | Reading, manipulating and analysing data in DataFrames                 |
| **scipy.stats** (`ss`)          | Statistics and tests                           |
| **matplotlib.pyplot** (`plt`)   | Creation of static graphics                                           |
| **seaborn**                     | High-level statistical visualisation                                  |
| **scikit-learn**                | train_test_split  , StandardScaler , SelectKBest     , train_test_split, División de datos en traintest  ,SandardScale,  Escalado de variables numéricas,SelectKBest, f_classif    , Selección univariante de features (ANOVA) ,RFE, Selección recursiva de features ,LogisticRegression   ,    KNeighborsClassifier,DecisionTreeClassifier,RandomForestClassifier,GradientBoostingClassifier,GridSearchCV,cross_val_score,confusion_matrix     |
| **imblearn**                    |            SMOTE                     |
| **XGBoost**                     |         XGBClassifier             |                    
| **LightGBM**                    |         LGBMClassifier              |              
| **CatBoost**                    |        CatBoostClassifier          |        
| **Useful**                  |bootcampviztools (`src/utils/bootcampviztools`) ,toolbox (`src/utils/toolbox`),sys.path.append("./src/utils")  |




**🗂️Repository content**
------------

```
├── main.ipynb # **Clean Notebook with code**
├── Presentacion_ML_Adicion_redes_sociales.pdf # Presentation used to explain the work realized
├── src/ # Folder with additional archives
│   └── data_sample 
        └── Students_Social_Media_Addiction # Dataset used
    └── img
    └── models
    └── notebooks
    └── utils
├── README.md # This file :)


## README (ES)

**ML Adicciones a redes sociales**
------------
ML_Adicciones_a_redes_sociales es un analisis de Machine Learning desarrollado como parte de un trabajo individual en el bootcamp de Data Science en The Bridge, Julio 2025.

**🧭Tabla de contenidos**
-----------

  - [Objetivo](#objetivo)
  - [Dataset utilizado](#dataset-utilizado)
  - [Solución aportada](#solución-aportada)
  - [Librerias necesarias](#librerias-necesarias)
  - [Contenido del repositorio](#contenido-del-repositorio)


---

**🎯Objetivo**
-----------

Bienvenido a mi proyecto en el cual queremos analizar los datos de nuestro dataset e definir el mejor modelo de predicción del nivel de adiccion de los estudiantes a redes sociales. 


**📂Dataset utilizado**
-----------

Para realizar este modelo usamos un dataset de Kaggle (url: https://www.kaggle.com/code/adilshamim8/social-media-addiction-among-students/notebook), sobre addicciones de estudiatnes a la redes sociales, que tiene 13 columnas y  705 instancias. 

Este archivo CSV contiene respuestas a encuestas realizadas a estudiantes de entre 16 y 25 años de varios países, en las que se recogen sus patrones de uso de las redes sociales junto con indicadores clave personales, académicos y de bienestar. 

Incluye campos numericos y categóricos que permiten el análisis exploratorio y el modelado predictivo sobre las relaciones entre la adicción a las redes sociales y diversos resultados vitales.

**🧠Solución aportada**
-----------


El trabajo se inicio con el analisis del dataset y de las variables disponibles. 
Nos enfrentamos a un problema de clasificacion multiclase, no balanceado. 
Tras entenderlo se realizó un mini EDA de las variables (univariante y bivariantes) con visualizaciones varias disponibles desde Seaborn y Matplotlib. 

Luego se utilizaron varias metodologias de selección de features con fin de poder realizar un modelo mas eficiente.
Varios modelos (8) fueron comparados y finalemente se decidio optimizar el que mejor respondia a nuestras necesidades (Random Forest Classifier).

Seguidamente, predejimos nuestros set de test contra el mejor modelo y su mejor seleccion de features, y tras su mejor optimización se evaluo su resultado con una matriz de confusion. Lo resultados nos parecieron satisfactorios por el problema que teniamos asi que se guardó  el modelo y lo podeis encotrar aqui ('src/models'). 



**📦Librerias necesarias**
-----------
| Librería                        | Uso principal                                                            |
|---------------------------------|---------------------------------------------------------------------------|
| **numpy**                       | Cálculos numéricos y operaciones con arrays                              |
| **pandas**                      | Lectura, manipulación y análisis de datos en DataFrames                  |
| **scipy.stats** (`ss`)          | Estadísticos y tests                             |
| **matplotlib.pyplot** (`plt`)   | Creación de gráficos estáticos                                            |
| **seaborn**                     | Visualización estadística de alto nivel                                   |
| **scikit-learn**                | train_test_split  , StandardScaler , SelectKBest     , train_test_split, División de datos en traintest  ,SandardScale,  Escalado de variables numéricas,SelectKBest, f_classif    , Selección univariante de features (ANOVA) ,RFE, Selección recursiva de features ,LogisticRegression   ,    KNeighborsClassifier,DecisionTreeClassifier,RandomForestClassifier,GradientBoostingClassifier,GridSearchCV,cross_val_score,confusion_matrix     |
| **imblearn**                    |            SMOTE                     |
| **XGBoost**                     |         XGBClassifier             |                    
| **LightGBM**                    |         LGBMClassifier              |              
| **CatBoost**                    |        CatBoostClassifier          |        
| **Utilidades**                  |bootcampviztools (`src/utils/bootcampviztools`) ,toolbox (`src/utils/toolbox`),sys.path.append("./src/utils")  |



---

**🗂️Contenido del repositorio**
-----------

```
├── main.ipynb # **Notebook de codigo limpio**
├── Presentacion_ML_Adicion_redes_sociales.pdf # Presentacion usada para explicar el analisis realizado
├── src/ # Carpeta de material adiciones
│   └── data_sample 
        └── Students_Social_Media_Addiction # Dataset utilizado
    └── img
    └── models
    └── notebooks
    └── utils
├── README.md # Este archivo :)
```
