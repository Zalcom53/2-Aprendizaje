import random
from collections import Counter
import math

def entrena_arbol(datos, target, clase_default, 
                  max_profundidad=None, acc_nodo=1.0, min_ejemplos=0,
                  variables_seleccionadas=None):
    """
    Entrena un árbol de decisión para datos numéricos.
    
    Parámetros:
    - datos: Lista de diccionarios con los datos de entrenamiento
    - target: Nombre del atributo objetivo
    - clase_default: Clase por defecto para nodos hoja
    - max_profundidad: Profundidad máxima del árbol
    - acc_nodo: Precisión mínima para considerar un nodo como hoja
    - min_ejemplos: Número mínimo de ejemplos para dividir un nodo
    - variables_seleccionadas: Número de variables a considerar en cada nodo (para bosques aleatorios)
    """
    # Manejo de caso base: si no hay datos, retornar un nodo hoja
    if not datos:
        return NodoN(terminal=True, clase_default=clase_default)

    # Obtener lista de atributos, excluyendo el atributo objetivo
    atributos = list(datos[0].keys())
    atributos.remove(target)
    
    # Selección aleatoria de atributos para bosques aleatorios
    if variables_seleccionadas is not None and isinstance(variables_seleccionadas, int):
        atributos = random.sample(atributos, min(variables_seleccionadas, len(atributos)))
    
    # Criterios para determinar si es un nodo hoja
    if len(atributos) == 0 or len(datos) <= min_ejemplos:
        return NodoN(terminal=True, clase_default=clase_default)
    
    # Determinar la clase más común para usar como clase por defecto
    clases = Counter(d[target] for d in datos)
    clase_default = clases.most_common(1)[0][0]
    
    # Verificar si se alcanzó la profundidad máxima o si el nodo es suficientemente puro
    if (max_profundidad == 0 or 
        clases.most_common(1)[0][1] / len(datos) >= acc_nodo):
        return NodoN(terminal=True, clase_default=clase_default)
    
    # Seleccionar la mejor variable y valor para dividir
    variable, valor = selecciona_variable_valor(datos, target, atributos)
    nodo = NodoN(
        terminal=False, 
        clase_default=clase_default,
        atributo=variable, 
        valor=valor 
    )
    
    # Dividir los datos según la variable y valor seleccionados
    datos_menor = [d for d in datos if d[variable] < valor]
    datos_mayor = [d for d in datos if d[variable] >= valor]
    
    # Si la división no es efectiva, retornar un nodo hoja
    if not datos_menor or not datos_mayor:
        return NodoN(terminal=True, clase_default=clase_default)
    
    # Recursivamente construir los subárboles
    nodo.hijo_menor = entrena_arbol(
        datos_menor,
        target,
        clase_default,
        max_profundidad - 1 if max_profundidad is not None else None,
        acc_nodo, min_ejemplos, variables_seleccionadas
    )   
    nodo.hijo_mayor = entrena_arbol(
        datos_mayor,
        target,
        clase_default,
        max_profundidad - 1 if max_profundidad is not None else None,
        acc_nodo, min_ejemplos, variables_seleccionadas
    )   
    return nodo

def selecciona_variable_valor(datos, target, atributos):
    """
    Selecciona la mejor variable y valor para dividir los datos.
    """
    entropia = entropia_clase(datos, target)
    mejor = max(
        ((a, maxima_ganancia_informacion(datos, target, a, entropia))
            for a in atributos),
        key=lambda x: x[1][1]
    )
    return mejor[0], mejor[1][0]

def entropia_clase(datos, target):
    """
    Calcula la entropía de la clase objetivo en los datos.
    """
    clases = Counter(d[target] for d in datos)
    total = sum(clases.values())
    return -sum((c/total) * math.log2(c/total) for c in clases.values())

def maxima_ganancia_informacion(datos, target, atributo, entropia):
    """
    Encuentra el punto de división que maximiza la ganancia de información para un atributo.
    """
    lista_valores = [(d[atributo], d[target]) for d in datos]
    lista_valores.sort(key=lambda x: x[0])
    lista_valor_ganancia = []
    for (v1, v2) in zip(lista_valores[:-1], lista_valores[1:]):
        if v1[1] != v2[1]:
            valor = (v1[0] + v2[0]) / 2
            ganancia = ganancia_informacion(datos, target, atributo, valor, entropia)
            lista_valor_ganancia.append((valor, ganancia))
    return max(lista_valor_ganancia, key=lambda x: x[1]) if lista_valor_ganancia else (lista_valores[0][0], 0)

def ganancia_informacion(datos, target, atributo, valor, entropia):
    """
    Calcula la ganancia de información para un atributo y un valor de división.
    """
    datos_menor = [d for d in datos if d[atributo] < valor]
    datos_mayor = [d for d in datos if d[atributo] >= valor]
    
    if not datos_menor or not datos_mayor:
        return 0
    
    entropia_menor = entropia_clase(datos_menor, target)
    entropia_mayor = entropia_clase(datos_mayor, target)
    
    total = len(datos)
    total_menor = len(datos_menor)
    total_mayor = len(datos_mayor)
    
    return (
        entropia 
        - (total_menor / total) * entropia_menor 
        - (total_mayor / total) * entropia_mayor 
    )

class NodoN:
    """
    Representa un nodo en el árbol de decisión para datos numéricos.
    """
    def __init__(self, terminal, clase_default, atributo=None, valor=None):
        self.terminal = terminal
        self.clase_default = clase_default
        self.atributo = atributo
        self.valor = valor
        self.hijo_menor = None
        self.hijo_mayor = None
    
    def predice(self, instancia):
        """
        Realiza una predicción para una instancia dada.
        """
        if self.terminal:
            return self.clase_default               
        if instancia[self.atributo] < self.valor:
            return self.hijo_menor.predice(instancia)       
        return self.hijo_mayor.predice(instancia)

def predice_arbol(arbol, datos):
    """
    Realiza predicciones para un conjunto de datos usando el árbol de decisión.
    """
    return [arbol.predice(d) for d in datos]

def evalua_arbol(arbol, datos, target):
    """
    Evalúa el rendimiento del árbol de decisión en un conjunto de datos.
    """
    predicciones = predice_arbol(arbol, datos)
    return sum(1 for p, d in zip(predicciones, datos) if p == d[target]) / len(datos)

def imprime_arbol(nodo, nivel=0):
    """
    Imprime la estructura del árbol de decisión.
    """
    if nodo.terminal:
        print("    " * nivel + f"La clase es {nodo.clase_default}")
    else:
        print("    " * nivel + f"Si {nodo.atributo} < {nodo.valor} entonces:")
        imprime_arbol(nodo.hijo_menor, nivel + 1)
        print("    " * nivel + f"Si {nodo.atributo} >= {nodo.valor} entonces:")
        imprime_arbol(nodo.hijo_mayor, nivel + 1)