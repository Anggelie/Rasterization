import os
import numpy as np

class Mesh:
    def __init__(self):

        self.faces = []
        self.albedo = None
        self.albedo_path = None
        self.normalmap = None
        self.normalmap_path = None

def _load_mtl(mtl_path):
    tex_for = {}
    if not os.path.exists(mtl_path):
        return tex_for
    cur = None
    with open(mtl_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            if not line.strip() or line.lstrip().startswith('#'):
                continue
            parts = line.strip().split()
            if parts[0] == 'newmtl':
                cur = parts[1]
                tex_for[cur] = None
            elif parts[0] == 'map_Kd' and cur:
                tex_for[cur] = " ".join(parts[1:])
    return tex_for

def _compute_tbn(p0, p1, p2, uv0, uv1, uv2, n_hint=None):
    # vectores de arista en espacio objeto
    e1 = p1 - p0
    e2 = p2 - p0
    # deltas UV
    duv1 = uv1 - uv0
    duv2 = uv2 - uv0
    denom = duv1[0]*duv2[1] - duv2[0]*duv1[1]
    if abs(denom) < 1e-8:
        N = np.cross(e1, e2)
        if np.linalg.norm(N) < 1e-8:
            N = np.array([0.0, 0.0, 1.0], np.float32)
        N = N / (np.linalg.norm(N) + 1e-8)
        T = np.array([1.0, 0.0, 0.0], np.float32)
        if abs(np.dot(T, N)) > 0.9:
            T = np.array([0.0, 1.0, 0.0], np.float32)
    else:
        f = 1.0 / denom
        T = f * (duv2[1] * e1 - duv1[1] * e2)
        B = f * (-duv2[0] * e1 + duv1[0] * e2)
        N = np.cross(e1, e2) if n_hint is None else n_hint

    N = N / (np.linalg.norm(N) + 1e-8)
    T = T - N * np.dot(N, T)
    T_norm = np.linalg.norm(T)
    if T_norm < 1e-8:
        T = np.array([1.0, 0.0, 0.0], np.float32)
        if abs(np.dot(T, N)) > 0.9:
            T = np.array([0.0, 1.0, 0.0], np.float32)
    else:
        T = T / T_norm
    B = np.cross(N, T)
    B = B / (np.linalg.norm(B) + 1e-8)

    return np.column_stack([T.astype(np.float32), B.astype(np.float32), N.astype(np.float32)])

def load_obj(path):
    V, VT, VN = [], [], []
    faces = []
    mtl_map = {}
    current_tex = None

    base = os.path.dirname(path)

    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            if not line.strip() or line.lstrip().startswith('#'):
                continue
            parts = line.strip().split()
            tag = parts[0]

            if tag == 'mtllib':
                mtl_file = os.path.join(base, " ".join(parts[1:]))
                mtl_map = _load_mtl(mtl_file)

            elif tag == 'usemtl':
                name = " ".join(parts[1:])
                tex = mtl_map.get(name)
                if tex:
                    current_tex = os.path.join(base, tex)

            elif tag == 'v':
                V.append(np.array(list(map(float, parts[1:4])), dtype=np.float32))
            elif tag == 'vt':
                u, v = map(float, parts[1:3])
                VT.append(np.array([u, 1.0 - v], dtype=np.float32))  # flip V
            elif tag == 'vn':
                VN.append(np.array(list(map(float, parts[1:4])), dtype=np.float32))

            elif tag == 'f':
                verts = parts[1:]
                tri_sets = [verts[0:3]] if len(verts) < 4 else [verts[0:3], [verts[0], verts[2], verts[3]]]

                for tri in tri_sets:
                    p, t, n = [], [], []
                    for token in tri:
                        a = token.split('/')
                        vi = int(a[0]) - 1 if a[0] else 0
                        ti = int(a[1]) - 1 if len(a) > 1 and a[1] else None
                        ni = int(a[2]) - 1 if len(a) > 2 and a[2] else None

                        p.append(V[vi])
                        t.append(VT[ti] if ti is not None and ti < len(VT) else np.array([0.0, 0.0], np.float32))
                        n.append(VN[ni] if ni is not None and ni < len(VN) else np.array([0.0, 0.0, 1.0], np.float32))

                    p0, p1, p2 = p[0], p[1], p[2]
                    uv0, uv1, uv2 = t[0], t[1], t[2]
                    n_hint = (n[0] + n[1] + n[2]) / 3.0
                    TBN = _compute_tbn(p0, p1, p2, uv0, uv1, uv2, n_hint)

                    faces.append({'p': p, 't': t, 'n': n, 'TBN': TBN})

    mesh = Mesh()
    mesh.faces = faces
    mesh.albedo_path = current_tex
    return mesh
