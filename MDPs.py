"""
Clases y funciones para definir y resolver MDPs discretos
utilizando programación dinámica y Q-learning en forma tabular.

"""
from abc import ABCMeta, abstractmethod
from random import choice, random

"""
Clases y funciones para definir y resolver MDPs discretos
utilizando programación dinámica y, potencialmente, Q-learning en forma tabular.

Incluye:
- Clase base MDP (abstracta)
- Iteración de política
- Iteración de valor
- Función para calcular Q(s, a)
- Utilidad para resolver con un solo método
- Utilidad para imprimir políticas
"""

from abc import ABCMeta, abstractmethod
from random import choice, random

class MDP(metaclass=ABCMeta):
    """
    Clase abstracta para definir un MDP discreto.
    
    Atributos esperados:
    - self.estados : conjunto de estados
    - self.gama : factor de descuento

    Métodos que deben implementarse:
    - acciones_legales(s)
    - recompensa(s, a, s')
    - prob_transicion(s, a, s')
    - es_terminal(s)
    """
    def __init__(self, gama, rho, meta):
        self.gama = gama
        self.rho = rho
        self.meta = meta
        self.estados = tuple(range(1, meta + 2))  # estados desde 1 hasta meta+1

    def acciones_legales(self, s):
        if self.es_terminal(s):
            return []
        return ['caminar', 'usar_camion']

    def recompensa(self, s, a, s_):
        if s_ > self.meta:
            return -9  # penalización por pasarse
        elif s_ == self.meta:
            return 0   # recompensa neutra por llegar justo a la meta
        elif a == 'caminar':
            return -1  # costo por caminar
        else:  # 'usar_camion'
            return -2  # costo por usar el camión

    def prob_transicion(self, s, a, s_):
        if self.es_terminal(s):
            return 1 if s_ == s else 0  # estado terminal: no cambia
        elif a == 'caminar':
            siguiente = min(s + 1, self.meta + 1)
            return 1 if s_ == siguiente else 0
        elif a == 'usar_camion':
            avance = min(2 * s, self.meta + 1)
            if s_ == avance:
                return self.rho
            elif s_ == s:
                return 1 - self.rho
            else:
                return 0

    def es_terminal(self, s):
        return s == self.meta

def valor_politica(pi, mdp, epsilon=1e-6, max_iter=1000):
    V = {s: 0 for s in mdp.estados}
    for _ in range(max_iter):
        delta = 0
        for s in mdp.estados: 
            if not mdp.es_terminal(s):
                v = V[s]
                V[s] = sum(
                    mdp.prob_transicion(s, pi[s], s_) 
                    * (mdp.recompensa(s, pi[s], s_) + mdp.gama * V[s_])
                    for s_ in mdp.estados
                )
                delta = max(delta, abs(v - V[s]))
        if delta < epsilon:
            break
    return V

def iteracion_politica(mdp, epsilon=1e-6, max_iter=1000):
    pi = {s: choice(mdp.acciones_legales(s)) 
          for s in mdp.estados if not mdp.es_terminal(s)}
    
    for _ in range(max_iter):
        V = valor_politica(pi, mdp, epsilon, max_iter)
        estable = True
        for s in mdp.estados:
            if not mdp.es_terminal(s):
                a = pi[s]
                mejor_a = max(
                    mdp.acciones_legales(s),
                    key=lambda a: sum(
                        mdp.prob_transicion(s, a, s_) 
                        * (mdp.recompensa(s, a, s_) + mdp.gama * V[s_])
                        for s_ in mdp.estados
                    )
                )
                pi[s] = mejor_a
                if a != mejor_a:
                    estable = False
        if estable:
            break
    return pi

def iteracion_valor(mdp, epsilon=1e-6, max_iter=1000, ver_V=False, debug=False):
    V = {s: 0 if mdp.es_terminal(s) else random() for s in mdp.estados}
    
    for i in range(max_iter):
        delta = 0
        for s in mdp.estados:
            if not mdp.es_terminal(s):
                v = V[s]
                V[s] = max(
                    sum(
                        mdp.prob_transicion(s, a, s_) 
                        * (mdp.recompensa(s, a, s_) + mdp.gama * V[s_])
                        for s_ in mdp.estados
                    )
                    for a in mdp.acciones_legales(s)
                )
                delta = max(delta, abs(v - V[s]))
        if debug and i % 100 == 0:
            print(f"Iteración {i + 1} - Delta: {delta}")
        if delta < epsilon:
            break
    
    pi = {
        s: max(
            mdp.acciones_legales(s),
            key=lambda a: sum(
                mdp.prob_transicion(s, a, s_) 
                * (mdp.recompensa(s, a, s_) + mdp.gama * V[s_])
                for s_ in mdp.estados
            )
        )
        for s in mdp.estados if not mdp.es_terminal(s)
    }
    
    return (pi, V) if ver_V else pi

def q_estado(s, a, V, mdp):
    """
    Calcula el valor Q(s, a) dado el valor de los estados V y un MDP.
    """
    return sum(
        mdp.prob_transicion(s, a, s_) *
        (mdp.recompensa(s, a, s_) + mdp.gama * V[s_])
        for s_ in mdp.estados
    )

def resolver_mdp(mdp, metodo='valor', **kwargs):
    """
    Resuelve un MDP utilizando el método especificado.
    
    Parámetros:
    - mdp : objeto MDP
    - metodo : 'valor' o 'politica'
    
    kwargs adicionales se pasan al método correspondiente.
    """
    if metodo == 'valor':
        return iteracion_valor(mdp, **kwargs)
    elif metodo == 'politica':
        return iteracion_politica(mdp, **kwargs)
    else:
        raise ValueError("Método no soportado. Usa 'valor' o 'politica'.")

def imprimir_politica(pi):
    """
    Imprime la política en formato legible.
    """
    print("Política óptima:")
    for s in sorted(pi):
        print(f"  Estado {s} → Acción: {pi[s]}")

    

def valor_politica(pi, mdp, epsilon=1e-6, max_iter=1000):
    """
    Calcula el valor de una política pi para un MDP.
    
    Parámetros
    ----------
    pi : dict
        Política pi que asigna a cada estado una acción.
    mdp : MDP
        MDP para el que se calcula el valor de la política.
    epsilon : float
        Criterio de convergencia.
    max_iter : int
        Número máximo de iteraciones.
        
    Devuelve
    --------
    V : dict
        Valor de la política pi.
    
    """
    V = {s: 0 for s in mdp.estados}
    
    for _ in range(max_iter):
        delta = 0
        for s in mdp.estados: 
            if not mdp.es_terminal(s):
                v = V[s]
                V[s] = sum(
                    mdp.prob_transicion(s, pi[s], s_) 
                    * (mdp.recompensa(s, pi[s], s_) + mdp.gama * V[s_])
                    for s_ in mdp.estados
                )
                delta = max(delta, abs(v - V[s]))
        if delta < epsilon:
            break
    return V

def iteracion_politica(mdp, epsilon=1e-6, max_iter=1000):
    """
    Calcula la política óptima para un MDP utilizando iteración de política.
    
    Parámetros
    ----------
    mdp : MDP
        MDP para el que se calcula la política óptima.
    epsilon : float
        Criterio de convergencia.
    max_iter : int
        Número máximo de iteraciones.
        
    Devuelve
    --------
    pi : dict
        Política óptima.
    
    """
    pi = {s: choice(mdp.acciones_legales(s)) 
          for s in mdp.estados if not mdp.es_terminal(s)}
    
    for _ in range(max_iter):
        V = valor_politica(pi, mdp, epsilon, max_iter)
        optima = True
        for s in mdp.estados:
            if not mdp.es_terminal(s):
                a = pi[s]
                pi[s] = max(
                    mdp.acciones_legales(s),
                    key=lambda a: sum(
                        mdp.prob_transicion(s, a, s_) 
                        * (mdp.recompensa(s, a, s_) + mdp.gama * V[s_])
                    for s_ in mdp.estados
                    )
                )
                if a != pi[s]:
                    estable = False
        if estable:
            break
    return pi

def iteracion_valor(mdp, epsilon=1e-6, max_iter=1_000, debug=False):
    """
    Calcula la política óptima para un MDP utilizando iteración de valor.
    
    Parámetros
    ----------
    mdp : MDP
        MDP para el que se calcula la política óptima.
    epsilon : float
        Criterio de convergencia.
    max_iter : int
        Número máximo de iteraciones.
    debug : bool
        Si es True, imprime el valor de delta cada 100 iteraciones.
        
    Devuelve
    --------
    pi : dict
        Política óptima.
    
    """
    V = {s: 0 for s in mdp.estados}
    
    for it in range(max_iter):
        delta = 0
        for s in mdp.estados:
            if not mdp.es_terminal(s):
                v = V[s]
                V[s] = max(
                    sum(
                        mdp.prob_transicion(s, a, s_) 
                        * (mdp.recompensa(s, a, s_) + mdp.gama * V[s_])
                        for s_ in mdp.estados
                    )
                    for a in mdp.acciones_legales(s)
                )
                delta = max(delta, abs(v - V[s]))
        if debug and it % 50 == 0:
            print(f"Iteración {it + 1} - Delta: {delta}")
        if delta < epsilon:
            break
    
    pi = {s: max(
        mdp.acciones_legales(s),
        key=lambda a: sum(
            mdp.prob_transicion(s, a, s_) 
            * (mdp.recompensa(s, a, s_) + mdp.gama * V[s_])
            for s_ in mdp.estados
        )
    ) for s in mdp.estados if not mdp.es_terminal(s)}
    
    return pi, V
    
    