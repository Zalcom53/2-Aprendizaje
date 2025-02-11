import utileria as ut
import bosque_aleatorio as ba
import random
import os

# Configuración del conjunto de datos Wine
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data'
archivo = 'datos/wine.data'
atributos = ['class', 'Alcohol', 'Malic acid', 'Ash', 'Alcalinity of ash', 'Magnesium', 
             'Total phenols', 'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins', 
             'Color intensity', 'Hue', 'OD280/OD315 of diluted wines', 'Proline']
target = 'class'

# Crear directorio de datos si no existe
if not os.path.exists('datos'):
    os.makedirs('datos')

# Descargar datos si no existen localmente
if not os.path.exists(archivo):
    ut.descarga_datos(url, archivo)

# Cargar datos
datos = ut.lee_csv(archivo, atributos=atributos, separador=',')

# Convertir atributos numéricos a float
for d in datos:
    for attr in atributos:
        if attr != target:
            d[attr] = float(d[attr])

# Mezclar los datos aleatoriamente
random.seed(42)
random.shuffle(datos)

# Dividir en conjunto de entrenamiento (80%) y prueba (20%)
N = int(0.8 * len(datos))
datos_entrenamiento = datos[:N]
datos_prueba = datos[N:]

def experimento(num_arboles, max_profundidad, variables_por_nodo):
    """
    Realiza un experimento con el bosque aleatorio y devuelve las precisiones.
    """
    bosque = ba.entrena_bosque_aleatorio(datos_entrenamiento, target, num_arboles, 
                                         max_profundidad=max_profundidad, 
                                         variables_por_nodo=variables_por_nodo)
    acc_entrenamiento = ba.evalua_bosque(bosque, datos_entrenamiento, target)
    acc_prueba = ba.evalua_bosque(bosque, datos_prueba, target)
    return acc_entrenamiento, acc_prueba

# Experimento 1: Variando el número de árboles
print("Experimento 1: Variando el número de árboles")
for num_arboles in [1, 5, 10, 50, 100]:
    acc_train, acc_test = experimento(num_arboles, max_profundidad=10, variables_por_nodo=None)
    print(f"Num. árboles: {num_arboles}, Acc. entrenamiento: {acc_train:.4f}, Acc. prueba: {acc_test:.4f}")

# Experimento 2: Variando la profundidad máxima
print("\nExperimento 2: Variando la profundidad máxima")
for max_profundidad in [1, 3, 5, 10, None]:
    acc_train, acc_test = experimento(50, max_profundidad=max_profundidad, variables_por_nodo=None)
    print(f"Max. profundidad: {max_profundidad}, Acc. entrenamiento: {acc_train:.4f}, Acc. prueba: {acc_test:.4f}")

# Experimento 3: Variando el número de variables por nodo
print("\nExperimento 3: Variando el número de variables por nodo")
for variables_por_nodo in [1, 3, 5, 7, None]:
    acc_train, acc_test = experimento(50, max_profundidad=10, variables_por_nodo=variables_por_nodo)
    print(f"Variables por nodo: {variables_por_nodo}, Acc. entrenamiento: {acc_train:.4f}, Acc. prueba: {acc_test:.4f}")