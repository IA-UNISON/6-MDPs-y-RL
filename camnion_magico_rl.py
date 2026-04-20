"""
El camión mágico, pero ahora por simulación

"""

from RL import MDPsim, SARSA, Q_learning, PoliticaGreedy
from random import random, randint

class CamionMagico(MDPsim):
    """
    Clase que representa un MDP para el problema del camión mágico.
    
    Si caminas, avanzas 1 con coso 1
    Si usas el camion, con probabilidad rho avanzas el doble de donde estabas
    y con probabilidad 1-rho te quedas en el mismo lugar. Todo con costo 2.
    
    El objetivo es llegar a la meta en el menor costo posible
    
    """    
    
    def __init__(self, gama, rho, meta):
        self.gama = gama
        self.rho = rho
        self.meta = meta
        self.estados = tuple(range(1, meta + 1))
    
    def estado_inicial(self):
        #return randint(1, self.meta // 2 + 1)
        #return randint(1, self.meta - 1)
        return 1
    
    def acciones_legales(self, s):
        return (['caminar', 'usar_camion'] if s < self.meta // 2 else
                ['caminar'] if s < self.meta else 
                [])
    
    def recompensa(self, s, a, s_):
        return  -1  if a == 'caminar' else -2 
        
    def transicion(self, s, a):
        if a == 'caminar':
            return s + 1
        elif a == 'usar_camion':
            return 2*s if random() < self.rho else s
        
    def es_terminal(self, s):
        return s >= self.meta

mdl = CamionMagico(gama=0.999, rho=0.999, meta=145)
    
Q_sarsa = SARSA(mdl, epsilon=0.2, alfa=0.5,  n_ep=5_000, n_iter=150)
pi_s = PoliticaGreedy(Q_sarsa)

Q_ql = Q_learning(mdl, epsilon=0.2, alfa=0.5,  n_ep=5_000, n_iter=150)
pi_ql = PoliticaGreedy(Q_ql)

print(f"Los tramos donde se debe usar el camión segun SARSA son:")
print([s for s in mdl.estados if pi_s(s) == 'usar_camion'])
print("-"*50)
print(f"Los tramos donde se debe usar el camión segun Qlearning son:")
print([s for s in mdl.estados if pi_ql(s) == 'usar_camion'])
print("-"*50)


"""
**********************************************************************************
Ahora responde a las siguientes preguntas:
**********************************************************************************

- Prueba con diferentes valores de rho. ¿Qué observas? ¿Porqué crees que pase eso?
Siendo 0.9 el valor default de rho, al cambiarlo a 0.1, los tramos tomados fueron muy
grandes. Por ejemplo, con 0.9 los primeros dos pasos segun Sarsa fueron 2 y 4,
con 0.1 fueron 11 y 35. Si se cambia el valor a 2, fueron 2, y 3. Esto se debe a que
el si el Camion Magico funciona o no es inversamente proporcional al valor de rho. Sin
embargo, el que los saltos de valores sean mayores cuando rho es menor se debe a que el
agente espera a estar más cerca de la meta antes de arriesgarse a usar el camión, con el
fin de maximizar el impacto del salto en caso de que funcione.

- Prueba con diferentes valores de gama. ¿Qué observas? ¿Porqué crees que pase eso?
Siendo su valor default de 0.999, al cambiarlo a 0.009 no hay resultados, y al cambiarlo a
2 se obtuvieron muchos resultados, donde el valor maximo sacado por SARSA fue de 107, y el
de Qlearning de 106. Esto se debe a que gama es que tanto valor el agente los valores obtenidos.
Al ser este de un valor muy bajo, no acepta ninguno y por lo tanto no hubo resultados, siendo
el caso opuesto al ser un valor muy alto.

- ¿Qué tan diferente es la política óptima de SARSA y Q-learning?
Qlearning parece priorizar mas el uso del Camion Magico a comparacion de SARSA, por lo que este
es mas optimo, ya que SARSA va por un paso mas seguro pero lento, al alternar entre ambos camiones.

- ¿Cambia mucho el resultado cambiando los valores de recompensa?
Por como se explico antes, si, pues estos afectan el comportamiento del agente, aunque tales cambios
son proporcionales.

- ¿Cuantas iteraciones se necesitan para que funcionen correctamente los algoritmos?
Esto depende de Tamaño de los Estados, que tanto se Explora y el Factor de Descuento. Aunque con
100,000 Estados para las 50 iteraciones en SARSA y 1000 en Qlearning de este codigo se tienen suficientes.

- ¿Qué pasaria si ahora el estado inicial es cualquier estado de la mitad para abajo?
Pues para toda meta n, seria empezar desde un estado entre 1 a n/2, haciendo que la meta sea un valor aleatorio
1 a meta-1, y su desempeño variaria de acuerdo a que tan bajo empiece.
**********************************************************************************

"""