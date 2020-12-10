#!/usr/bin/env python
# coding: utf-8

# ---
# 
# ## Universidad de Costa Rica
# ### Escuela de Ingeniería Eléctrica
# #### IE0405 - Modelos Probabilísticos de Señales y Sistemas
# 
# Segundo semestre del 2020
# 
# ---
# 
# * Estudiante: **Fabricio Alpízar Rodríguez**
# * Carné: **B60275**
# * Grupo: **01**
# 
# 
# # `P5` - Análisis y predicción del consumo diario de energía
# 
# > Esta actividad extiende el análisis y modelado realizados anteriormente sobre la demanda eléctrica del país a partir de una base de datos del Instituto Costarricense de Electricidad, del 2019. El estudio se orienta ahora en el uso de las cadenas de Markov para determinar la probabilidad de ocurrencia de múltiples estados para el consumo energético diario.
# 
# ---
# * Elaboración de nota teórica y demostración: **Jeaustin Sirias Chacón**, como parte de IE0499 - Proyecto Eléctrico: *Estudio y simulación de aplicaciones de la teoría de probabilidad en la ingeniería eléctrica*.
# * Revisión: **Fabián Abarca Calderón**
# 

# ---
# ## 1. - El último vals: *Las cadenas de Markov*
# 
# En el proyecto programado anterior (`P4` - Modulación digital IQ) se estudiaron los **procesos estocásticos**. Algunos de estos tienen la **propiedad de Markov**, según la cual se puede decir que provoca en el sistema una especie de "amnesia" al momento de determinar *valores futuros* y condiciona la determinación de probabilidades a partir **únicamente** de los valores presentes.

# ## 2. - Construyendo un modelo energético con cadenas de Markov
# 
# Con la previa reseña es posible intentar construir un modelo básico para el consumo diario nacional de energía en Costa Rica durante el 2019 con las cadenas de Markov. Para ello se reutilizará la base de datos de demanda energética también empleada en los proyectos programados P2 y P3.
# 
# ### 2.1 - Funciones a implementar 
# 
# A continuación se especificarán las funciones auxiliares a desarrollar, para la construcción de la cadena de Markov:
# 
# 1. `energia_diaria(archivo_json)`: Importa la base de datos completa en formato **JSON** y calcula la energía diaria usando [la regla del trapecio](https://es.wikipedia.org/wiki/Regla_del_trapecio) y retorna un vector con el valor de energía de cada día.
# 
# 2. `definicion_estados(vector_energia, numero_estados)`: Clasifica a cada valor de energía en el rango de 1 a `numero_estados` según el nivel de energía y retorna un vector con cada estado.
# 
# 3. `probabilidad_transicion(vector_estados, numero_estados, presente, futuro)`: Calcula la probabilidad de transición entre un estado inicial $i$ en $t$ y un estado futuro $j$ en $t+1$. Retorna la probabilidad $\Pi_{i,j}$ de transición entre $i$ y $j$, donde:
# 
# \begin{equation}
# P(X_{t+1} = j \mid X_{t} = i) = \Pi_{i,j}
# \end{equation}

# #### 2.1.1 - Calculando el consumo diario y parámetros relevantes con `energia_diaria`

# In[27]:


import pandas as pd
import numpy as np
from datetime import datetime

def energia_diaria(archivo_json):
    '''Importa la base de datos completa y devuelve
    un vector con la energía diaria, en MWh.
    
    :param archivo_json: el contenedor con datos crudos
    :return: el vector de energía diaria
    '''
    # Cargar el "DataFrame"
    df = pd.read_json(archivo_json) 

    # Convertir en un array de NumPy
    datos = np.array(df)  

    # Crear vector con todos los valores horarios de demanda
    demanda = []

    # Extraer la magnitud de la demanda para todas las horas
    for hora in range(len(datos)):
        demanda.append(datos[hora][0]['MW'])

    # Separar las magnitudes en grupos de 24 (24 h)
    demanda = np.split(np.array(demanda), len(demanda) / 24)

    # Crear vector para almacenar la energía a partir de la demanda
    energia = []

    # Calcular la energía diaria por la regla del trapecio
    for dia in range(len(demanda)):
        E = round(np.trapz(demanda[dia]), 2)
        energia.append(E)

    return energia 


# #### 2.1.2 -  Definiendo el número de estados de energía con `definir_estados`

# In[4]:


import numpy as np

def definicion_estados(vector_energia, estados):
    '''Una función que se encarga de retornar
    los límites del rango de energía para
    una cantidad arbitraria de estados sobre 
    la base del vector de energía.
    
    :param energia: vector de energía diaria
    :param estados: el número de estados
    :return: el vector de estados
    '''
    
    minimo = np.min(vector_energia)
    maximo = np.max(vector_energia)
    segmento = (maximo - minimo)/estados
    vector_estados = np.empty(len(vector_energia))
    
    for i, dia in enumerate(vector_energia):
        diferencia = dia - minimo
        proporcion = diferencia // segmento
        vector_estados[i] = proporcion + 1
        
    return vector_estados


# #### 2.1.3 - Calculando la ocurrencia de las transiciones por estado con `calcular_transiciones`
# 
# El objetivo de la función será retornar el número de ocurrencias (y por tanto la frecuencia relativa) de la transición de un estado presente $i$ a un estado próximo $j$; es decir, retorna puntualmente una probabilidad $\Pi_{ij}$  de transición entre las muchas que puede contener la **matriz de transición** $\Pi$ en función de sus $N$ estados. Obsérvese la siguiente matriz de estados generalizada:
# 
# 
# $$
# \Pi = \begin{bmatrix}
# \Pi_{11} & \ldots & \Pi_{1N} \\ 
# \Pi_{21}& \ldots & \Pi_{2N}\\ 
# \vdots& \ddots & \vdots\\ 
# \Pi_{N1} & \ldots & \Pi_{NN}
# \end{bmatrix}
# $$
# 
# Puesto que $\Pi$ siempre es una **matriz cuadrada**, entonces habrá $N^2$ probabilidades de transición dentro de la misma. Ahora, dado a que la función `calcular_transiciones` retorna solo una de estas probabilidades, **por ejecución**.
# 
# **Nota**: ¿Qué ocurriría si se analiza un proceso con $N=10$ estados? Evidentemente, no sería práctico ejecutar esta función $10^2$ veces para completar la matriz $\Pi$.

# In[5]:


import numpy as np

def probabilidad_transicion(vector_estados, numero_estados, presente, futuro):
    '''Una función que se encarga de calcular
    la probabilidad de ocurrencia de la transición
    entre un estado inicial 'i' y un estado futuro 'j'.
    
    :param vector_estados: el vector con los todos los estados
    :param presente: el número del estado presente
    :param futuro: el número del estado futuro
    :return: la probabilidad de transición
    '''
    
    # Recorrer el vector_estados
    ocurrencias_i = 0
    ocurrencias_i_j = 0
    for i, estado in enumerate(vector_estados[0:-1]):
        if estado == presente:
            ocurrencias_i += 1
            if vector_estados[i+1] == futuro:
                ocurrencias_i_j += 1
    
    # Cálculo de la probabilidad
    probabilidad = ocurrencias_i_j / ocurrencias_i
    
    return probabilidad


# ## 3. - Demostración de las funciones implementadas

# In[6]:


import matplotlib.pyplot as plt

# Importar los datos y calcular la energía diaria
vector_energia = energia_diaria('demanda_2019.json')

# Definir los estados
numero_estados = 10
vector_estados = definicion_estados(vector_energia, numero_estados)
print(vector_estados)

# Graficar la evolución de los estados
plt.plot(vector_estados)
plt.xlabel('Día del año')
plt.ylabel('Consumo de energía (estado)')
plt.show()

# Definir la probabilidad de transición de "i" a "j"
i, j = 10, 9
Pi_ij = probabilidad_transicion(vector_estados, numero_estados, i, j)
print('Pi_ij =', Pi_ij)


# ---
# ## 4. - Asignaciones del proyecto
# 
# ### Asignación de parámetros
# 
# Las asignaciones requieren de valores de $t$, $i$, $j$ asignados según carné. 

# In[37]:


from numpy import random
from scipy import stats

def parametros_asignados(digitos):
    '''Elige un valor t aleatoriamente,
    dos estados arbitrarios i y j
    '''
    
    random.seed(digitos)
    estados = [i+1 for i in range(10)]
    T = stats.expon(2)
    t = int(T.rvs())
    i = estados[random.randint(0, len(estados))]
    j = estados[random.randint(0, len(estados))]
    print('t: {}, i: {}, j: {}'.format(t, i, j))
    return t, i, j


# **Ejemplo**: el carné B12345 utiliza los dígitos 12345 y obtiene los parámetros $t$: 4, $i$: 2, $j$: 5.

# In[38]:


t, i, j = parametros_asignados(60275)


# ### 4.1. - Encuesta del curso
# 
# * (30%) Completar la encuesta disponible a partir del lunes 7 de diciembre de 2020.
# 
# ### 4.2. - Construir la matriz de transición de estados
# 
# * (30%) Para los datos dados, crear la matriz de probabilidades de transición de estados, considerando que el rango de valores de energía se divide en $N = 10$ estados, donde $i, j \in \{ 1, \ldots, 10 \}$ son estados particulares.

# In[23]:


from numpy import random
from scipy import stats
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np

# Importar los datos y calcular la energía diaria
vector_energia = energia_diaria("demanda_2019.json")

# Definir los estados
numero_estados = 10
vector_estados = definicion_estados(vector_energia, numero_estados)

# Inicialización de variable que almacena las filas de la matriz deseada
matriz = []
    
# Determinación de la probabilidad de transición del estado i al estado j
for i in range(1, 11):
    fila =[]
    for j in range(1, 11):
        pi_j = probabilidad_transicion(vector_estados, numero_estados, i, j)
        fila.append(pi_j)
    matriz.append(fila)
    
T = np.matrix(matriz)

# Se presenta la Matriz

# Lo siguiente es simplemente para presentar la matriz T bien ordenada y que se vea bien
print("Matriz de probabilidades de transición de estados, para N = {} Estados: ".format(numero_estados),"\n")
v1 = '      '
v2 = '  '
for n, e in enumerate(T):
    print('[', end="")
    s=0
    for s in range(10):
        x = str(round(T[n,s],5))
        if s < 9: 
            if len(x)==7:
                print(round(T[n,s], 5),v2,end="")
            else:
                print(round(T[n,s], 5),v1,end="")
        else:
            print(round(T[n,s], 5),end="")
            if len(x)==7:
                print(']')
            else:
                print('    ]')


# ### 4.3. - Construcción de la matriz de transición de orden *t* predicción
# 
# Para los valores obtenidos en `parametros_asignados()`:
# 
# * (20%) Construir la matriz de transición de estados de orden $t$.

# In[53]:


from numpy import random
from scipy import stats
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np 


print("Parámetros asignados al carné B60275: ")
t,i,j =  parametros_asignados(60275)


# Se procede a construir la matriz de transición de estados de orden t
M = T

for a in range(t-1): 
       if t > 1:
            M = np.dot(M,T)
            
print("\n")
# Lo siguiente es simplemente para presentar la matriz T bien ordenada y que se vea bien
print("Matriz de transición de estados de orden {}:".format(t))
v1 = '      '
v2 = '  '
v3 = '   '
for n, e in enumerate(T):
    print('[', end="")
    s=0
    for s in range(10):
        x = str(round(M[n,s],5))
        if s < 9: 
            if len(x)==7:
                print(round(M[n,s], 5),v2,end="")
            else:
                if len(x)==6:
                    print(round(M[n,s], 5),v3,end="")
                else:
                    print(round(M[n,s], 5),v1,end="")
            
        else:
            print(round(M[n,s], 5),end="")
            if len(x)==7:
                print(']')
            else:
                print('    ]')


# * (20%) Determinar la probabilidad de estar en el estado $j$, $t$ días después de estar en el estado $i$.

# In[43]:


from numpy import random
from scipy import stats
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np

print("La probabilidad de estar en el estado {}, {} días después de estar en el estado {}, es: ".format(j, t, i))
print("Pi_j:", round((M[i-1,j-1]),6))


# ---
# 
# ### Universidad de Costa Rica
# #### Facultad de Ingeniería
# ##### Escuela de Ingeniería Eléctrica
# 
# ---
