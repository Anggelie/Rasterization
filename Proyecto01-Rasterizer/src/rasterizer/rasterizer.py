import numpy as np


def _to_h(v3):
    """Convierte (x,y,z) a homogéneo (x,y,z,1)."""
    return np.array([v3[0], v3[1], v3[2], 1.0], dtype=np.float32)

def _normalize(v):
    n = np.linalg.norm(v)
    return v if n < 1e-8 else (v / n)

def _ndc_to_screen(ndc_xy, w, h):
    """
    Pasa de NDC [-1,1] a coordenada de pixel [0..w-1], [0..h-1]
    con origen en la esquina superior izquierda.
    """
    x = (ndc_xy[0] * 0.5 + 0.5) * (w - 1)
    y = (1.0 - (ndc_xy[1] * 0.5 + 0.5)) * (h - 1)  # y invertida
    return np.array([x, y], dtype=np.float32)

def _edge(a, b, c):
    """Producto cruzado 2D (signo sirve para test de dentro/fuera)."""
    return (c[0] - a[0]) * (b[1] - a[1]) - (c[1] - a[1]) * (b[0] - a[0])

def _bbox(pts, w, h):
    """AABB en pantalla, clampeado a los límites del framebuffer."""
    minx = int(max(0, np.floor(np.min(pts[:, 0]))))
    miny = int(max(0, np.floor(np.min(pts[:, 1]))))
    maxx = int(min(w - 1, np.ceil(np.max(pts[:, 0]))))
    maxy = int(min(h - 1, np.ceil(np.max(pts[:, 1]))))
    return minx, miny, maxx, maxy

def _sample_nearest(img, uv, flip_v=False):
    """Muestreo nearest de [0..1]x[0..1] -> RGB [0..1]."""
    if img is None:
        return np.array([1.0, 1.0, 1.0], dtype=np.float32)
    h, w = img.shape[0], img.shape[1]

    u = np.clip(uv[0], 0.0, 1.0) * (w - 1)
    if flip_v:
        v = (1.0 - np.clip(uv[1], 0.0, 1.0)) * (h - 1)
    else:
        v = np.clip(uv[1], 0.0, 1.0) * (h - 1)

    return img[int(v), int(u), :].astype(np.float32) / 255.0

def _perspective_interp(w0, w1, w2, a0, a1, a2, recip_w0, recip_w1, recip_w2):
    """
    Interpolación perspectiva-correcta para atributos 'a' (vec2/vec3).
    a_over_w = a * (1/w);  result = sum(wi * ai_over_w) / sum(wi * 1/wi)
    """
    num = w0 * a0 * recip_w0 + w1 * a1 * recip_w1 + w2 * a2 * recip_w2
    den = w0 * recip_w0 + w1 * recip_w1 + w2 * recip_w2 + 1e-12
    return num / den

def _depth_from_ndc(z_ndc):
    """
    Acepta z en [-1,1] (OpenGL-like) o [0,1] (DirectX-like).
    Devuelve z normalizado a [0,1] para el Z-buffer.
    """
    if -1.5 <= z_ndc <= 1.5:
        z = (z_ndc * 0.5) + 0.5
    else:
        z = z_ndc
    return float(np.clip(z, 0.0, 1.0))


# Rasterizer principal 

def draw_mesh(mesh, M, V, P, color_buf, z_buf, uniforms, fragment_shader):
    """
    Dibuja un mesh triángulo por triángulo con:
      - Transformación M (model), V (view), P (projection)
      - Interpolación perspectiva-correcta de posición, normal y UV
      - Z-buffer (float32, 0..1)
      - Normal mapping por-TBN si 'normalmap_img' está en uniforms
      - Baricéntricas en vary['bary'] (para wireframe/contornos en shaders)

    mesh.faces: lista de dicts con claves:
      'p' -> (3,3) posiciones
      't' -> (3,2) uvs (opcional)
      'n' -> (3,3) normales (opcional)
      'TBN' -> (3,3) matriz TBN por cara (opcional, identidad si no viene)
    """
    H, W = color_buf.shape[0], color_buf.shape[1]

    MV    = V @ M
    MVP   = P @ MV
    MinvT = np.transpose(np.linalg.inv(M))  # para transformar normales

    if "light_dir" in uniforms and uniforms["light_dir"] is not None:
        uniforms["light_dir"] = _normalize(
            uniforms["light_dir"].astype(np.float32)
        )

    flip_v_for_normals = bool(uniforms.get("flip_v", False))

    for face in mesh.faces:
        p_obj = np.asarray(face.get('p', []), dtype=np.float32)                 # (3,3)
        t_obj = np.asarray(face.get('t', [(0.0, 0.0)] * 3), dtype=np.float32)   # (3,2)
        n_obj = np.asarray(face.get('n', [(0.0, 0.0, 1.0)] * 3), dtype=np.float32)  # (3,3)
        TBN_face = np.asarray(face.get('TBN', np.eye(3, dtype=np.float32)), dtype=np.float32)

        if p_obj.shape != (3, 3):
            continue

        # Transformaciones por vértice
        p_world = np.empty((3, 3), dtype=np.float32)
        p_clip  = np.empty((3, 4), dtype=np.float32)
        n_world = np.empty((3, 3), dtype=np.float32)

        for i in range(3):
            pw = (M @ _to_h(p_obj[i]))[:3]
            p_world[i] = pw
            p_clip[i]  = MVP @ _to_h(p_obj[i])

            n4 = MinvT @ np.array([n_obj[i, 0], n_obj[i, 1], n_obj[i, 2], 0.0], dtype=np.float32)
            n_world[i] = _normalize(n4[:3])

        if np.all(p_clip[:, 3] <= 0.0):
            continue

        ndc = p_clip[:, :3] / p_clip[:, [3]]
        scr = np.empty((3, 2), dtype=np.float32)
        for i in range(3):
            scr[i] = _ndc_to_screen(ndc[i, :2], W, H)

        minx, miny, maxx, maxy = _bbox(scr, W, H)
        if minx > maxx or miny > maxy:
            continue

        recip_w = 1.0 / (p_clip[:, 3] + 1e-12)

        area = _edge(scr[0], scr[1], scr[2])
        if abs(area) < 1e-8:
            continue

        pos_over_w = p_world * recip_w[:, None]
        nrm_over_w = n_world * recip_w[:, None]
        uv_over_w  = t_obj   * recip_w[:, None]

        for y in range(miny, maxy + 1):
            for x in range(minx, maxx + 1):
                Pp = np.array([x + 0.5, y + 0.5], dtype=np.float32)

                w0 = _edge(scr[1], scr[2], Pp)
                w1 = _edge(scr[2], scr[0], Pp)
                w2 = _edge(scr[0], scr[1], Pp)

                # Mismo signo => punto dentro del triángulo
                if not ((w0 >= 0 and w1 >= 0 and w2 >= 0) or (w0 <= 0 and w1 <= 0 and w2 <= 0)):
                    continue

                # Baricéntricas normalizadas por el área del triángulo
                w0 /= area; w1 /= area; w2 /= area

                z_ndc = w0 * ndc[0, 2] + w1 * ndc[1, 2] + w2 * ndc[2, 2]
                z = _depth_from_ndc(z_ndc)
                if z >= z_buf[y, x]:
                    continue

                pos = _perspective_interp(w0, w1, w2,
                                          pos_over_w[0], pos_over_w[1], pos_over_w[2],
                                          recip_w[0], recip_w[1], recip_w[2])

                nrm = _perspective_interp(w0, w1, w2,
                                          nrm_over_w[0], nrm_over_w[1], nrm_over_w[2],
                                          recip_w[0], recip_w[1], recip_w[2])
                nrm = _normalize(nrm)

                uvp = _perspective_interp(w0, w1, w2,
                                          uv_over_w[0], uv_over_w[1], uv_over_w[2],
                                          recip_w[0], recip_w[1], recip_w[2])

                # Normal mapping opcional
                nm_img = uniforms.get("normalmap_img", None)
                if nm_img is not None:
                    tex_n = _sample_nearest(nm_img, uvp, flip_v_for_normals) * 2.0 - 1.0  # [-1,1]
                    nrm = _normalize(TBN_face @ tex_n)

                vary = {
                    "pos": pos,
                    "normal": nrm,
                    "uv": uvp,
                    # Baricéntricas para wireframe/contornos en shaders:
                    "bary": np.array([w0, w1, w2], dtype=np.float32),
                }

                # Sombreado
                col = fragment_shader(vary, uniforms)   # RGB [0..1]
                col = np.clip(col, 0.0, 1.0) * 255.0
                color_buf[y, x, :] = col.astype(np.uint8)
                z_buf[y, x] = z
