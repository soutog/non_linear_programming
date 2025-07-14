import scipy.optimize as opt
from math import pi

# solving facility location problem using direct integration with the solver

# DADOS DE ENTRADA
# coordenadas dos clientes (pontos destino) e demandas
customers = {
    1: {"x": 0.0, "y": 0.0, "d": 20},
    2: {"x": 10.0, "y": 0.0, "d": 25},
    3: {"x": 0.0, "y": 12.0, "d": 15},
    4: {"x": 10.0, "y": 12.0, "d": 10},
}

n = len(customers)  # numero de clientes
unit_cost = 1.0  # custo unitário de transporte
W = 2  # número de armazéns a instalar

# define a pattern for the decision variable as vector
# x = [x1, y1, x2, y2, ..., xW, yW, q11, q12, ..., qWN]
