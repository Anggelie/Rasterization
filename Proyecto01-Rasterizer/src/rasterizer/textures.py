from PIL import Image
import numpy as np
import os

def load_image(path):
    img = Image.open(path).convert("RGB")
    return np.asarray(img, dtype=np.uint8)

def load_texture(path):
    if not path or not os.path.exists(path): return None
    return load_image(path)

def load_normal_map(path):
    if not path or not os.path.exists(path): return None
    img = Image.open(path).convert("RGB")
    arr = np.asarray(img, dtype=np.float32) / 255.0
    n = arr * 2.0 - 1.0
    norm = np.linalg.norm(n, axis=2, keepdims=True)
    norm = np.clip(norm, 1e-6, None)
    return n / norm

def sample(texture, uv):
    if texture is None:
        return np.array([200,200,200], dtype=np.uint8)
    H,W = texture.shape[:2]
    u = (uv[0] % 1.0) * (W-1)
    v = (uv[1] % 1.0) * (H-1)
    x = int(u); y = int(v)
    return texture[y, x]

def sample_normal(normalmap, uv):
    if normalmap is None: return None
    H,W = normalmap.shape[:2]
    u = (uv[0] % 1.0) * (W-1)
    v = (uv[1] % 1.0) * (H-1)
    x = int(u); y = int(v)
    return normalmap[y, x]
