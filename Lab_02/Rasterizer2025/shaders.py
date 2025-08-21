import numpy as np

def vertexShader(vertex, modelMatrix=None):
    if modelMatrix:
        vertex = [sum(vertex[j] * modelMatrix[j][i] for j in range(3)) + modelMatrix[3][i] for i in range(3)]
    return vertex

def fragmentShader(color):
    return color
