import random
from arboles_numericos import entrena_arbol, NodoN

def entrena_bosque_aleatorio(datos, target, num_arboles, max_profundidad=None, variables_por_nodo=None):
    """
    Entrena un bosque aleatorio.
    
    Parámetros:
    - datos: Lista de diccionarios con los datos de entrenamiento
    - target: Nombre del atributo objetivo
    - num_arboles: Número de árboles en el bosque
    - max_profundidad: Profundidad máxima de cada árbol
    - variables_por_nodo: Número de variables a considerar en cada nodo
    """
    bosque = []
    # Determinar la clase por defecto (la más común en los datos)
    clase_default = max(set(d[target] for d in datos), key=lambda x: sum(1 for d in datos if d[target] == x))
    
    for _ in range(num_arboles):
        # Crear una muestra bootstrap de los datos
        datos_muestra = random.choices(datos, k=len(datos))
        # Entrenar un árbol con la muestra y agregarlo al bosque
        arbol = entrena_arbol(datos_muestra, target, clase_default, 
                              max_profundidad=max_profundidad, 
                              variables_seleccionadas=variables_por_nodo)
        bosque.append(arbol)
    
    return bosque

def predice_bosque(bosque, instancia):
    """
    Realiza una predicción para una instancia usando el bosque aleatorio.
    """
    # Obtener predicciones de todos los árboles
    predicciones = [arbol.predice(instancia) for arbol in bosque]
    # Retornar la predicción más común (voto mayoritario)
    return max(set(predicciones), key=predicciones.count)

def evalua_bosque(bosque, datos, target):
    """
    Evalúa el rendimiento del bosque aleatorio en un conjunto de datos.
    """
    aciertos = sum(1 for d in datos if predice_bosque(bosque, d) == d[target])
    return aciertos / len(datos)