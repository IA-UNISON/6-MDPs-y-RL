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
1.- Prueba con diferentes valores de rho. ¿Qué observas? ¿Porqué crees que pase eso?
    Si rho>0.5, el agente arriesga más porque es más probable ganar.
    Si rho<0.5, apuesta menos porque perder es más fácil.
    Si rho=0.5, el agente no arriesga nada porque no tiene sentido.
    Si rho=0, el agente no avanza nada porque no tiene sentido.

2.- Prueba con diferentes valores de gama. ¿Qué observas? ¿Porqué crees que pase eso?
    Si gamma es muy alto, el agente arriesga más porque le importa más el futuro.
    Si gamma es muy bajo, el agente arriesga menos porque le importa más el presente.
    Si gamma=0, el agente no arriesga nada porque no tiene sentido.

3.- ¿Qué tan diferente es la política óptima de SARSA y Q-learning?
    SARSA aprende según lo que realmente hace el agente.
    Q-learning aprende como si siempre tomara la mejor acción.
    SARSA suele ser más conservador y Q-learning más agresivo.

4.- ¿Cambia mucho el resultado cambiando los valores de recompensa?
    Sí, puede cambiar dependiendo de cómo se premien las acciones.
    Si la recompensa solo se da al llegar a la meta, el agente busca llegar ahí rápido.
    Si das más recompensas en el camino, puede que el agente prefiera otras rutas.
    Depende de qué tanto premies cada parte del juego.

6.- ¿Qué pasa si el costo de usar el camión es mayor que el de caminar?
    El juego sería más difícil.
    El agente empieza con menos dinero y más lejos de la meta.
    Entonces tiene más riesgo de perder, y probablemente apueste menos o le tome más tiempo aprender a ganar.
**********************************************************************************

"""