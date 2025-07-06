# %%
import pandas as pd
import numpy as np
import seaborn as sns
from scipy.stats import pearsonr

# %%
tips = sns.load_dataset("tips")
iris = sns.load_dataset("iris")
taxis = sns.load_dataset("taxis")
tips.head(5)

# %%
iris.head(5)

# %%
taxis.head(5)

# %%
taxis.info()

# %%
taxis.payment.value_counts()

# %% [markdown]
# 
# Funcion: describe_df
# Esta función debe recibir como argumento un dataframe y debe devolver una dataframe como el de la imagen (NO el de la imagen). Es decir, un dataframe que tenga una columna por cada variable del dataframe original, y como filas los tipos de dichas variables, el tanto por ciento de valores nulos o missings, los valores únicos y el porcentaje de cardinalidad.
# 
# La figura muestra el resultado esperado de llamar a la función con el dataset del Titanic:

# %%
def descripcion(df):
    descr = pd.DataFrame({
        'DATA_TYPE': df.dtypes,
        'MISSINGS (%)': (df.isnull().mean() * 100).round(2),
        'UNIQUE_VALUES': df.nunique(),
        'CARD (%)': (df.nunique() / len(df) * 100).round(2)})    
    return descr.T

# %%
descripcion(tips)

# %%
descripcion(iris)

# %%
descripcion(taxis)

# %% [markdown]
# Funcion: tipifica_variables
# Esta función debe recibir como argumento un dataframe, un entero (umbral_categoria) y un float (umbral_continua). 
# 
# La función debe devolver un dataframe con dos columnas "nombre_variable", "tipo_sugerido" que tendrá tantas filas como columnas el dataframe. En cada fila irá el nombre de una de las columnas y una sugerencia del tipo de variable. Esta sugerencia se hará siguiendo las siguientes pautas:
# 
# Si la cardinalidad es 2, asignara "Binaria"
# Si la cardinalidad es menor que umbral_categoria asignara "Categórica"
# Si la cardinalidad es mayor o igual que umbral_categoria, entonces entra en juego el tercer argumento:
# Si además el porcentaje de cardinalidad es superior o igual a umbral_continua, asigna "Numerica Continua"
# En caso contrario, asigna "Numerica Discreta"

# %%
def tipificacion_variables (df, umbral_cat = int(), umbral_continua = float()):
    df = pd.DataFrame([df.nunique(), df.nunique()/len(df) * 100, df.dtypes]) 
    df = df.T 
    df = df.rename(columns = {0: "Card", 1: "%_Card", 2: "Tipo"}) 

    df.loc[df.Card == 1, "%_Card"] = 0.00

    df["Tipo_sugerido"] = "Categorica"
    df.loc[df["Card"] == 2, "Tipo_sugerido"] = "Binaria"




    df.loc[df["Card"] >= umbral_cat, "Tipo_sugerido"] = "Numerica discreta"
    df.loc[df["%_Card"] >= umbral_continua, "Tipo_sugerido"] = "Numerica continua"
    

    return df[["Tipo_sugerido"]]

tipificacion_variables(tips, 10, 30)

# %%
tipificacion_variables(iris, 8, 20)

# %%
tipificacion_variables(taxis, 6, 50)

# %% [markdown]
# 
# Funcion: get_features_num_regression
# Esta función recibe como argumentos un dataframe,
#  el nombre de una de las columnas del mismo (argumento 'target_col'),que debería ser el target de un hipotético modelo de regresión, es decir debe ser una variable numérica continua o discreta pero con alta cardinalidad, además de un argumento 'umbral_corr', de tipo float que debe estar entre 0 y 1 y una variable float "pvalue" cuyo valor debe ser por defecto "None".
# 
# La función debe devolver una lista con las columnas numéricas del dataframe cuya correlación con la columna designada por "target_col" sea superior en valor absoluto al valor dado por "umbral_corr". Además si la variable "pvalue" es distinta de None, sólo devolvera las columnas numéricas cuya correlación supere el valor indicado y además supere el test de hipótesis con significación mayor o igual a 1-pvalue.
# 
# La función debe hacer todas las comprobaciones necesarias para no dar error como consecuecia de los valores de entrada. Es decir hará un check de los valores asignados a los argumentos de entrada y si estos no son adecuados debe retornar None y printar por pantalla la razón de este comportamiento. Ojo entre las comprobaciones debe estar que "target_col" hace referencia a una variable numérica continua del dataframe.

# %%
def get_features_num_regression(df, target_col, umbral_corr, pvalue=None):
    #comprobacion df
    if not isinstance(df, pd.DataFrame):
        print("El primer argumento no es un DataFrame")
        return None
    #comprobacion target_col
    if target_col not in df.columns:
        print("La columna target no pertenece a este DataFrame")
        return None
    #comprobacion umbral_corr
    if not (0 < umbral_corr < 1):
        print("Error: El umbral de correlacion debe ser entre 0 y 1")
    if not isinstance(umbral_corr, float):
        print("Error: El umbral de correlacion debe ser un foat")
        return None
    #comprobacion pvalue?
    
    #instancio la lista que sera mi return
    lista_numerica = []
    umbral_cat = 6

    for col in df: 
        if df[col].dtype == float:
            lista_numerica.append(col)
        if df[col].dtype == int:
             lista_numerica.append(col)
        if col == target_col:
            None
        else:
             None
        return lista_numerica

    #verifica que col target sea numerica continua o discreta con alta cardinalidad
    card_target = df[target_col].nunique()/len(df)*100
    if df[target_col].dtype != int and df[target_col].dtype != float:
        print("Error este target no es numerica")
    if card_target >= umbral_cat:
        print(f"El target es discreta: {card_target}")
    else:
        None
    
    corr = df.corr(numeric_only= True)
    corr_abs = np.abs(corr[target_col]).sort_values(ascending = False)
    for col2, correlacion in corr_abs.items(): 
        if correlacion > umbral_corr:
            lista_numerica.append(col2)
        else:
            None
        return lista_numerica

# %%
get_features_num_regression(tips, "tip", (0.2), pvalue=None)

# %%
get_features_num_regression(iris, "sepal_length", (0.7), pvalue=None)

# %%
get_features_num_regression(taxis, "fare", (0.1), pvalue=None)


