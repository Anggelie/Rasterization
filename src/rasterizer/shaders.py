import numpy as np

def _tex2d(img, uv):
    if img is None: 
        return np.array([1.0, 1.0, 1.0], np.float32)
    h, w = img.shape[0], img.shape[1]
    u = np.clip(uv[0], 0.0, 1.0) * (w - 1)
    v = np.clip(uv[1], 0.0, 1.0) * (h - 1)
    x0, y0 = int(u), int(v)
    return img[y0, x0, :].astype(np.float32) / 255.0

def phong_textured(varyings, uniforms):
    N = varyings['normal'] / (np.linalg.norm(varyings['normal']) + 1e-6)
    L = uniforms.get('light_dir', np.array([0.6,1.0,-0.5], np.float32))
    L = L / (np.linalg.norm(L) + 1e-6)
    V = uniforms.get('cam_pos', np.array([0,0,1], np.float32)) - varyings['pos']
    V = V / (np.linalg.norm(V) + 1e-6)

    albedo = _tex2d(uniforms.get('albedo_img'), varyings['uv'])

    ka = uniforms.get('ka', 0.35)
    kd = uniforms.get('kd', 0.9)
    ks = uniforms.get('ks', 0.25)
    shin = uniforms.get('shininess', 32.0)

    diff = max(np.dot(N, L), 0.0)
    H = (L + V); H = H / (np.linalg.norm(H) + 1e-6)
    spec = max(np.dot(N, H), 0.0) ** shin

    color = albedo * (ka + kd * diff) + spec * ks
    return np.clip(color, 0.0, 1.0)

def phong_shader(*args, **kwargs):
    pass
