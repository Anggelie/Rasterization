from math import sin, cos, radians

class Model:
    def __init__(self):
        self.vertices = []
        self.colors = []
        self.vertexShader = None
        self.fragmentShader = None
        self.translation = [0, 0, 0]
        self.rotation = [0, 0, 0]
        self.scale = [1, 1, 1]

    def GetModelMatrix(self):
        rx = radians(self.rotation[0])
        ry = radians(self.rotation[1])
        rz = radians(self.rotation[2])
        sx, sy, sz = self.scale
        tx, ty, tz = self.translation

        return [
            [sx * cos(rz), -sy * sin(rz), 0],
            [sx * sin(rz), sy * cos(rz), 0],
            [0, 0, sz],
            [tx, ty, tz],
        ]
