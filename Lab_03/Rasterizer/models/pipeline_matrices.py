import numpy as np

def get_model_matrix(translation, rotation, scale):
    T = np.identity(4)
    T[:3, 3] = translation

    rx, ry, rz = rotation
    rx, ry, rz = np.radians([rx, ry, rz])

    Rx = np.array([
        [1, 0, 0, 0],
        [0, np.cos(rx), -np.sin(rx), 0],
        [0, np.sin(rx), np.cos(rx), 0],
        [0, 0, 0, 1]
    ])

    Ry = np.array([
        [np.cos(ry), 0, np.sin(ry), 0],
        [0, 1, 0, 0],
        [-np.sin(ry), 0, np.cos(ry), 0],
        [0, 0, 0, 1]
    ])

    Rz = np.array([
        [np.cos(rz), -np.sin(rz), 0, 0],
        [np.sin(rz), np.cos(rz), 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])

    R = Rx @ Ry @ Rz
    S = np.diag([scale[0], scale[1], scale[2], 1])
    return T @ R @ S

def get_view_matrix(eye, target, up):
    z = np.array(eye) - np.array(target)
    z = z / np.linalg.norm(z).astype(np.float64)
    x = np.cross(up, z)
    x = x / np.linalg.norm(x).astype(np.float64)
    y = np.cross(z, x)

    mat = np.identity(4)
    mat[0, :3] = x
    mat[1, :3] = y
    mat[2, :3] = z
    mat[:3, 3] = -mat[:3, :3] @ eye
    return mat

def get_projection_matrix(fov, aspect, near, far):
    f = 1 / np.tan(np.radians(fov) / 2)
    mat = np.zeros((4, 4))
    mat[0, 0] = f / aspect
    mat[1, 1] = f
    mat[2, 2] = (far + near) / (near - far)
    mat[2, 3] = (2 * far * near) / (near - far)
    mat[3, 2] = -1
    return mat

def get_viewport_matrix(x, y, width, height):
    mat = np.identity(4)
    mat[0, 0] = width / 2
    mat[1, 1] = height / 2
    mat[0, 3] = x + width / 2
    mat[1, 3] = y + height / 2
    return mat
