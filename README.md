# ¿¿Mañana lloverá??

_Este proyecto es una solución del problema de "Rain in Australia" (https://www.kaggle.com/jsphyg/weather-dataset-rattle-package) de Machine Learning_

## Starting 🚀

_Estas instrucciones te ayudaran a copiar y correr el proyecto en tu computadora local._


```
git clone https://github.com/Quintaneishon/rain_tomorrow.git

cd rain_tomorrow
```

### Prerequisites 📋

* [Python 3.x](https://www.python.org/downloads/) - Lenguaje utilizado para esta solución.

## Ejecutando el proyecto ⚙️

```
pip install -r requirements.txt
python app.py
```

## Construido con 🛠️

* [pandas](https://pandas.pydata.org/) - Una herramienta opensource de análsis y manipulación de la información.
* [matplotlib](https://matplotlib.org/) - Es una librería para crear estaticas, animadas e interactivas visualizaciones en Python.
* [numpy](https://numpy.org/) - El paquete fundamental para la computación científica con Python.
* [seaborn](https://seaborn.pydata.org/) - Librería para visualizacion de la información en Python basada en matplotlib, provee una interfaz de alto nivel para dibujar gráficos estadísticos atractivos e informativos
* [sklearn](https://scikit-learn.org/stable/) - Herramientas sencillas y eficaces para el análisis predictivo de datos

## Observaciones 📑

La solución se compone de las siguientes partes:

1. LECTURA, ANÁLISIS Y LIMPIEZA DE LA INFORMACIÓN
2. TRATAMIENTO DE LA INFORMACIÓN PARA ML
3. MODELOS DE PREDICCIÓN

El paso 1 y 2 se realizaron sin ningun problema, con la experiencia quue tengo actualmente con el uso de Python y las librerías usadas.

El paso 3 de los modelos fue el que más trabajo me costo realizar por mi falta de experiencia en el tema de ML pero el único algoritmo utilizado ya lo había implementado en la solución de un reto en el Hackathon BBVA 2019

__Logistic Regression__: Es un algoritmo basado en la estadística que se utiliza en los problemas de clasificación. Permite predecir la probabilidad de que una entrada pertenezca a una determinada categoría.
Utiliza la función logit o la función sigmoide como núcleo.
![Regresion logistica](https://www.monografias.com/docs113/regresion-aplicada-logistica/image002.png)

Finalmente concluimos que: 

la solución tiene una **Accuracy Score de 0.8567029357421474**

## Author ✒️

* **Ajitzi Ricardo Quintana Ruiz** - *Ingeniero en Sistemas Computacionales del Instituto Politécnico Nacional* - [Quintaneishon](https://github.com/Quintaneishon)