import numpy as np

def vertexShader(vertex, modelMatrix, viewMatrix, projectionMatrix, viewportMatrix):
    v = np.array([*vertex, 1])
    return viewportMatrix @ projectionMatrix @ viewMatrix @ modelMatrix @ v

def fragmentShader(**kwargs):
    texCoords = kwargs.get("texCoords", [0, 0])
    texture = kwargs.get("texture", None)
    color = kwargs.get("color", [1, 1, 1])

    if texture is not None and isinstance(texture, np.ndarray):
        h, w = texture.shape[:2]
        u = int(texCoords[0] * (w - 1))
        v = int((1 - texCoords[1]) * (h - 1))  # invertido eje Y
        if 0 <= u < w and 0 <= v < h:
            return (texture[v, u] / 255).tolist()

    return color
