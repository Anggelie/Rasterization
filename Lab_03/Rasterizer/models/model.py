import numpy as np
from math import sin, cos, radians

class Model:
    def __init__(self, obj_path):
        from models.OBJ import load_obj_with_mtl  # se importa aquí para evitar loops
        self.vertices = []
        self.texcoords = []
        self.colors = []
        self.texture = None
        self.vertexShader = None
        self.fragmentShader = None
        self.translation = [0, 0, 0]
        self.rotation = [0, 0, 0]
        self.scale = [1, 1, 1]

        # Cargar desde OBJ
        verts, faces, materials, tex = load_obj_with_mtl(obj_path)

        from pygame import image
        from pygame.surfarray import array3d

        # Cargar texturas por material
        textures = {}
        for name, mat in materials.items():
            if 'map_Kd' in mat:
                path = mat['map_Kd']
                try:
                    textures[name] = array3d(image.load(path).convert())
                except:
                    print(f"No se pudo cargar: {path}")
                    textures[name] = None

        # Centrar y escalar
        verts_np = np.array(verts)
        min_vals = verts_np.min(axis=0)
        max_vals = verts_np.max(axis=0)
        center = (min_vals + max_vals) / 2
        size = np.max(max_vals - min_vals)
        s = 5.0 / size

        for face, mat_name in faces:
            self.texture = textures.get(mat_name, None)

            if len(face) >= 3:
                for j in range(1, len(face) - 1):
                    for idx in [face[0], face[j], face[j + 1]]:
                        vx, vy, vz = verts[idx[0]]
                        sx = (vx - center[0]) * s
                        sy = (vy - center[1]) * s
                        sz = (vz - center[2]) * s
                        self.vertices.extend([sx, sy, sz])

                        if idx[1] is not None and idx[1] < len(tex):
                            self.texcoords.extend(tex[idx[1]])
                        else:
                            self.texcoords.extend([0, 0])

                    self.colors.append([1, 1, 1])  # Color blanco por triángulo

    def GetModelMatrix(self):
        rx = radians(self.rotation[0])
        ry = radians(self.rotation[1])
        rz = radians(self.rotation[2])
        sx, sy, sz = self.scale
        tx, ty, tz = self.translation

        S = np.array([
            [sx, 0, 0, 0],
            [0, sy, 0, 0],
            [0, 0, sz, 0],
            [0, 0, 0, 1]
        ])

        Rx = np.array([
            [1, 0, 0, 0],
            [0, cos(rx), -sin(rx), 0],
            [0, sin(rx), cos(rx), 0],
            [0, 0, 0, 1]
        ])

        Ry = np.array([
            [cos(ry), 0, sin(ry), 0],
            [0, 1, 0, 0],
            [-sin(ry), 0, cos(ry), 0],
            [0, 0, 0, 1]
        ])

        Rz = np.array([
            [cos(rz), -sin(rz), 0, 0],
            [sin(rz), cos(rz), 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])

        T = np.array([
            [1, 0, 0, tx],
            [0, 1, 0, ty],
            [0, 0, 1, tz],
            [0, 0, 0, 1]
        ])

        R = Rx @ Ry @ Rz
        return T @ R @ S
