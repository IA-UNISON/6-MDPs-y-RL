"""
El problema del jugador pero como un problema de aprendizaje por refuerzo

"""

from RL import MDPsim, SARSA, Q_learning, PoliticaGreedy
from random import random, randint

class Jugador(MDPsim):
    """
    Clase que representa un MDP para el problema del jugador.
    
    El jugador tiene un capital inicial y el objetivo es llegar a un capital
    objetivo o quedarse sin dinero.
    
    """
    def __init__(self, meta, ph, gama):
        self.estados = tuple(range(meta + 1))
        self.meta = meta
        self.ph = ph
        self.gama = gama
        
    def estado_inicial(self):
        return randint(1, self.meta - 1)
    
    def acciones_legales(self, s):
        if s == 0 or s == self.meta:
            return []
        return [i for i in range(1, min(s, self.meta - s) + 1)]
    
    def recompensa(self, s, a, s_):
        return self.meta if s_ == self.meta else 0
    
    def transicion(self, s, a):
        return s + a if random() < self.ph else s - a
    
    def es_terminal(self, s):
        return s == 0 or s == self.meta
    
mdl = Jugador(meta=100, ph=0.40, gama=1)

Q_sarsa = SARSA( mdl, alfa=0.2, epsilon=0.02, n_ep=10_000, n_iter=100)
pi_s = PoliticaGreedy(Q_sarsa)

Q_ql = Q_learning( mdl, alfa=0.2, epsilon=0.02, n_ep=10_000, n_iter=100)
pi_q = PoliticaGreedy(Q_ql)

print("Estado".center(10) + '|' +  "SARSA".center(10) + '|' + "Q-learning".center(10))
print("-"*10 + '|' + "-"*10 + '|' + "-"*10)
for s in mdl.estados:
    if not mdl.es_terminal(s):
        print(str(s).center(10) + '|' 
              + str(pi_s(s)).center(10) + '|' 
              + str(pi_q(s)).center(10))
print("-"*10 + '|' + "-"*10 + '|' + "-"*10)

""" 
***************************************************************************************
Responde las siguientes preguntas:
***************************************************************************************
1. ¿Qué pasa si se modifica el valor de epsilón de la política epsilon-greedy?
Si el valor de epsilon es alto, el agente tomara muchas pero ineficientes acciones aleatorias,
y vicebersa si es bajo. Si es 0, no tomara decisiones.

2. ¿Para que sirve usar una politica epsilon-greedy?
Sirve para tomar decisiones de acuerdo a la recompensa que se nos da por cada una atravez
de refuerso.

3. ¿Qué pasa con la política óptima y porqué si p_h es mayor a 0.5?
Siendo p_h la probabilidad de ganar una apuesta, entonces, en promedio, apostar sera
favorable para el jugador.

4. ¿Y si es 0.5?
El juego se vuelve justo, pues el agente estara apostando y no por igual, aunque como consecuencia,
el valor aprendido podria no variar lo suficiente.

5. ¿Y si es menor a 0.5?
El juego se vuelve desfavorable, pues apostar suele ser mas riesgoso y la politica implementada
sera mas cautiva en cuanto a cuando apostar.

6. ¿Qué pasa si se modifica el valor de la tasa de aprendizaje?
Si el valor es grande, el agente se vuelve olvidadizo y el apendisaje inestable. Por ende, si es
bajo, tanto el agente como el aprendisaje se vuelven lentos. Un valor promedio equilibra estos
resultados.

7. ¿Qué pasa si se modifica el valor de gama?
Mientras menor el valor, mas priorizara el agente los resultados inmediatos. Mientras mayor, el
agente priorizara los resultados proximos en vez de los inmediatos.

***************************************************************************************

"""