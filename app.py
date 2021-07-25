# Primero importamos todas las librerias para la limpieza, visualizacion y procesamiento de la informacion
# pandas para la lectura y limpieza de los datos
import pandas as pd
# numpy para las operaciones aritmeticas
import numpy as np
# matplotlib para la visualizacion de la informacion
import matplotlib.pyplot as plt
# seaborn para graficar visualmente la informacion
import seaborn as sns
# libreria para ML
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
# importamos nuestras operaciones para que este codigo este mas limpio
from operations import *

FILE_NAME = "weatherAUS.csv"

"""
LECTURA, ANALISIS Y LIMPIEZA DE LA INFORMACION
"""
# convertimos los datos en un DataFrame
df = pd.read_csv(FILE_NAME)

# veamos un poco de la informacion del df
print(df.head())
print(df.shape)
print(df.info())

# veamos un resumen de la informacion 
# informacion medible ( puros numeros )
num_data = df.select_dtypes(exclude='object')
print(num_data.describe())

# inforacion con diferentes valores pero repetidos
obj_data = df.select_dtypes(include='object')
print(obj_data.describe())

# eliminamos las filas que sean nulas
quitar_nulos(df)
# veamos los valores posibles para las columnas que nos interesan
print(len(df['Location'].unique()))
print(len(df['Date'].unique()))
# vemos que la columna Date tiene muchos posibles valores, hay que corregir esto, convirtiendolo en datetime y separando por año, mes y dia
df['Date'] = pd.to_datetime(df['Date'])
df['year'] = df['Date'].dt.year
df['month'] = df['Date'].dt.month
df['day'] = df['Date'].dt.day
df.drop('Date', axis = 1, inplace = True)

print(df.shape)
print(df.head())

# vamos a encontrar valores que puedan afectar nuestro procesamiento, valores que pueden ser error al estar muy lejos de los demas, usaremos el algoritmo visual de Box plot
for col in num_data.columns:
    sns.boxplot(x=df[col])
    plt.show()

col_defectos = ['MinTemp', 'MaxTemp', 'Rainfall', 'Evaporation', 'WindGustSpeed','WindSpeed9am', 'WindSpeed3pm', 'Humidity9am', 'Pressure9am', 'Pressure3pm', 'Temp9am', 'Temp3pm']
remove_defectos(df, col_defectos)

# analicemos la columna que mas nos interesa la de RainTomorrow
names = df['RainTomorrow'].unique()
values = df['RainTomorrow'].value_counts()
plt.barh(names, values)
for index, value in enumerate(values):
    plt.text(value, index, str(value))
plt.show()

# analicemos visualmente la relacion entre sunshine vs rainfall y sunshine vs evaporation
sns.lineplot(data=df,x='Sunshine',y='Rainfall',color='red')
plt.show()
sns.lineplot(data=df,x='Sunshine',y='Evaporation',color='blue')
plt.show()

"""
EMPIEZA EL TRATAMIENTO DE LA INFORMACION PARA ML
"""

# necesitamos convertir nuestras columnas objeto a valores numericos, porque es necesariopara su tratamiento.
df['RainToday'].replace({'No':0, 'Yes': 1}, inplace = True)
df['RainTomorrow'].replace({'No':0, 'Yes': 1}, inplace = True)
df['WindGustDir'].replace(convert_obj_to_num(df,'WindGustDir'),inplace = True)
df['WindDir9am'].replace(convert_obj_to_num(df,'WindDir9am'),inplace = True)
df['WindDir3pm'].replace(convert_obj_to_num(df,'WindDir3pm'),inplace = True)
df['Location'].replace(convert_obj_to_num(df,'Location'), inplace = True) 

# Observemos la correlacion de nuestros datos ya limpios
plt.figure(figsize=(20,20))
sns.heatmap(df.corr(), linewidths=0.5, annot=False, fmt=".2f", cmap = 'viridis')
plt.show()

X = df.drop(['RainTomorrow'],axis=1)
y = df['RainTomorrow']

# Creamos nuestros trenes de entrenamiento y de pruebas
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 0)
print(f"Datos de entrenamiento: {len(X_train)}")
print(f"Datos de prueba: {len(X_test)}")

# normalzamos nuestros trenes
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

"""
AQUI EMPIEZAN LOS MODELOS DE PREDICCION
"""
# Regresión logística
classifier_logreg = LogisticRegression(solver='liblinear', random_state=0)
classifier_logreg.fit(X_train, y_train)
y_pred = classifier_logreg.predict(X_test)
print(y_pred)
# Evaluamos nuestro modelo
print("Accuracy Score: {}".format(accuracy_score(y_test,y_pred)))
print("Train Data Score: {}".format(classifier_logreg.score(X_train, y_train)))
print("Test Data Score: {}".format(classifier_logreg.score(X_test, y_test)))