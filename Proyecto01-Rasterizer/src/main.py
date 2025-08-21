# Universidad del Valle de Guatemala
# UVG – Proyecto 01 Rasterizer
# Anggelie Velásquez 221181


import os, sys, numpy as np
import math

BASE_DIR = os.path.dirname(os.path.abspath(__file__))   # .../src
ROOT_DIR = os.path.dirname(BASE_DIR)                    # .../
ASSETS   = os.path.join(ROOT_DIR, "assets")
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

try:
    from rasterizer.math3d import mat4_identity, translation, scale, rotation_y, rotation_x, rotation_z
    from rasterizer.camera   import Camera
    from rasterizer.obj      import load_obj
    from rasterizer.textures import load_texture, load_normal_map
    from rasterizer.rasterizer import draw_mesh
    from rasterizer.gl import load_background_to_buffer, save_bmp
    from rasterizer import shaders as RSH
except ModuleNotFoundError:
    sys.path.insert(0, ROOT_DIR)
    from src.rasterizer.math3d import mat4_identity, translation, scale, rotation_y, rotation_x, rotation_z
    from src.rasterizer.camera   import Camera
    from src.rasterizer.obj      import load_obj
    from src.rasterizer.textures import load_texture, load_normal_map
    from src.rasterizer.rasterizer import draw_mesh
    from src.rasterizer.gl import load_background_to_buffer, save_bmp
    from src import shaders as RSH

class AssetManager:
    def __init__(self, root_dir, assets_dir):
        self.root_dir = root_dir
        self.assets_dir = assets_dir
        self.models = {}
        self.textures = {}
        self.normalmaps = {}
        self.scan_assets()

    def scan_assets(self):
        print("[AssetManager] Escaneando assets...")
        scan_dirs = [
            self.root_dir,
            self.assets_dir,
            os.path.join(self.assets_dir, "models"),
            os.path.join(self.assets_dir, "textures"),
            os.path.join(self.assets_dir, "normalmaps"),
        ]
        for nm in ["cat", "Pikachu", "ojo", "girasol", "craneo"]:
            scan_dirs += [
                os.path.join(self.root_dir, nm),
                os.path.join(self.assets_dir, "models", nm),
            ]
        for d in scan_dirs:
            if os.path.exists(d):
                self._scan_directory(d)
        print(f"[AssetManager] Encontrados: {len(self.models)} modelos, {len(self.textures)} texturas, {len(self.normalmaps)} normal maps")

    def _scan_directory(self, directory):
        for root, _, files in os.walk(directory):
            rl = root.lower()
            for f in files:
                fp = os.path.join(root, f)
                fl = f.lower()
                if fl.endswith(".obj"):
                    self.models[self._model_key(fl, rl)] = fp
                elif fl.endswith((".png",".jpg",".jpeg",".bmp",".tga")):
                    if any(h in fl for h in ["normal","_n.","_norm","normalmap"]):
                        self.normalmaps[self._tex_key(fl, rl)] = fp
                    else:
                        self.textures[self._tex_key(fl, rl)] = fp

    def _model_key(self, fn, dr):
        if "cat" in fn or "cat" in dr: return "cat"
        if "pikachu" in fn or "pikachu" in dr: return "pikachu"
        if "eye" in fn or "ojo" in dr: return "eye"
        if "girasol" in fn or "sunflower" in fn or "girasol" in dr: return "sunflower"
        if "craneo" in fn or "skull" in fn or "craneo" in dr: return "skull"
        return os.path.splitext(fn)[0]

    def _tex_key(self, fn, dr):
        dr = dr.lower(); fn = fn.lower()
        if "cat" in dr or "cat" in fn or "garfield" in fn: return "cat"
        if "pikachu" in dr or "pikachu" in fn: return "pikachu"
        if "ojo" in dr or "eye" in dr or "eye" in fn or "iris" in fn: return "eye"
        if "girasol" in dr or "girasol" in fn or "sunflower" in fn: return "sunflower"
        if "craneo" in dr or "skull" in dr or "skull" in fn: return "skull"
        return os.path.splitext(fn)[0]

    def get_model(self, key):     return self.models.get(key)
    def get_texture(self, key):   return self.textures.get(key)
    def get_normalmap(self, key): return self.normalmaps.get(key)

assets = AssetManager(ROOT_DIR, ASSETS)


def _tex2d(img, uv):
    """Función mejorada para muestrear texturas con mejor manejo de errores"""
    if img is None:
        return np.array([0.8, 0.8, 0.8], np.float32)
    
    try:
        h, w = img.shape[0], img.shape[1]
        u = np.clip(uv[0], 0.0, 1.0) * (w-1)
        v = np.clip(uv[1], 0.0, 1.0) * (h-1)
        u_int, v_int = int(u), int(v)
        
        # Asegurar que los índices estén dentro de rango
        u_int = max(0, min(u_int, w-1))
        v_int = max(0, min(v_int, h-1))
        
        color = img[v_int, u_int, :].astype(np.float32) / 255.0
        return color
    except:
        return np.array([0.6, 0.6, 0.6], np.float32)

def advanced_toon_shader(vary, uni):
    """Shader toon avanzado con outlines y múltiples bandas de luz"""
    N = vary['normal']
    N /= (np.linalg.norm(N) + 1e-6)
    L = uni.get('light_dir', np.array([0.6, 1.0, -0.5], np.float32))
    L /= (np.linalg.norm(L) + 1e-6)
    V = uni.get('cam_pos', np.zeros(3, np.float32)) - vary['pos']
    V /= (np.linalg.norm(V) + 1e-6)
    
    albedo = _tex2d(uni.get('albedo_img'), vary['uv'])
    diff = max(np.dot(N, L), 0.0)
    
    # Múltiples niveles de toon shading
    if diff > 0.9:
        toon_factor = 1.8
    elif diff > 0.7:
        toon_factor = 1.4
    elif diff > 0.4:
        toon_factor = 1.0
    elif diff > 0.2:
        toon_factor = 0.6
    else:
        toon_factor = 0.3
    
    rim = 1.0 - max(np.dot(N, V), 0.0)
    rim_color = np.array([0.2, 0.4, 1.0], np.float32) * (rim ** 3) * 0.8
    
    base_color = albedo * toon_factor
    final_color = base_color + rim_color
    
    return np.clip(final_color, 0, 1)

def neon_wireframe_shader(vary, uni):
    """Shader wireframe neón que respeta la textura base"""
    u, v = vary['uv'][0], vary['uv'][1]
    pos = vary['pos']
    N = vary['normal']
    N /= (np.linalg.norm(N) + 1e-6)
    L = uni.get('light_dir', np.array([0.6, 1.0, -0.5], np.float32))
    L /= (np.linalg.norm(L) + 1e-6)
    
    albedo = _tex2d(uni.get('albedo_img'), vary['uv'])
    
    diff = max(np.dot(N, L), 0.0)
    base_color = albedo * (0.4 + 0.6 * diff)
    
    grid_size = 12.0
    line_width = 0.04
    
    grid_u = abs((u * grid_size) % 1.0 - 0.5) < line_width
    grid_v = abs((v * grid_size) % 1.0 - 0.5) < line_width
    
    glow_intensity = math.sin(pos[0] * 3.0) * 0.2 + 0.8
    
    if grid_u or grid_v:
        wire_color = np.array([0.0, 1.0, 1.0], np.float32) * glow_intensity
        return np.clip(base_color * 0.3 + wire_color * 0.7, 0, 1)
    else:
        tech_tint = np.array([0.9, 1.0, 1.1], np.float32)
        return np.clip(base_color * tech_tint, 0, 1)

def metallic_shader(vary, uni):
    """Shader metálico con reflexiones y anisotropía"""
    N = vary['normal']
    N /= (np.linalg.norm(N) + 1e-6)
    L = uni.get('light_dir', np.array([0.6, 1.0, -0.5], np.float32))
    L /= (np.linalg.norm(L) + 1e-6)
    V = uni.get('cam_pos', np.zeros(3, np.float32)) - vary['pos']
    V /= (np.linalg.norm(V) + 1e-6)
    u, v = vary['uv'][0], vary['uv'][1]
    
    base_color = np.array([0.9, 0.7, 0.2], np.float32)
    albedo = _tex2d(uni.get('albedo_img'), vary['uv'])
    metallic_base = base_color * 0.7 + albedo * 0.3
    
    diff = max(np.dot(N, L), 0.0) * 0.3
    
    R = 2.0 * np.dot(N, L) * N - L
    spec = max(np.dot(R, V), 0.0) ** 16.0
    
    center_u, center_v = 0.5, 0.5
    dist_from_center = math.sqrt((u - center_u)**2 + (v - center_v)**2)
    aniso_rings = math.sin(dist_from_center * 40.0) * 0.3 + 0.7
    
    # Fresnel para bordes
    fresnel = max(0.1, 1.0 - max(np.dot(N, V), 0.0))
    
    # Combinar efectos
    final_color = (metallic_base * (0.4 + diff) + 
                   spec * np.array([1.0, 0.9, 0.6], np.float32) * 2.0 +
                   fresnel * np.array([1.0, 0.8, 0.4], np.float32) * 0.5) * aniso_rings
    
    return np.clip(final_color, 0, 1)

def advanced_checkerboard_shader(vary, uni):
    """Shader de tablero avanzado con efectos 3D"""
    u, v = vary['uv'][0], vary['uv'][1]
    N = vary['normal']
    N /= (np.linalg.norm(N) + 1e-6)
    L = uni.get('light_dir', np.array([0.6, 1.0, -0.5], np.float32))
    L /= (np.linalg.norm(L) + 1e-6)
    
    scale1 = 6.0
    scale2 = 24.0
    
    checker1 = (int(u * scale1) % 2 + int(v * scale1) % 2) % 2
    checker2 = (int(u * scale2) % 2 + int(v * scale2) % 2) % 2
    
    # Colores base
    if checker1 == 0:
        if checker2 == 0:
            base = np.array([0.95, 0.95, 0.95], np.float32)  
        else:
            base = np.array([0.85, 0.85, 0.9], np.float32)   
    else:
        if checker2 == 0:
            base = np.array([0.15, 0.15, 0.15], np.float32)  
        else:
            base = np.array([0.05, 0.05, 0.1], np.float32)  
    
    diff = max(np.dot(N, L), 0.0)
    shaded_color = base * (0.4 + 0.6 * diff)
    
    albedo = _tex2d(uni.get('albedo_img'), vary['uv'])
    final_color = shaded_color * 0.8 + albedo * 0.2
    
    return np.clip(final_color, 0, 1)

def galaxy_shader(vary, uni):
    """Shader de galaxia que respeta la textura del ojo"""
    u, v = vary['uv'][0], vary['uv'][1]
    pos = vary['pos']
    N = vary['normal']
    N /= (np.linalg.norm(N) + 1e-6)
    L = uni.get('light_dir', np.array([0.6, 1.0, -0.5], np.float32))
    L /= (np.linalg.norm(L) + 1e-6)
    
    albedo = _tex2d(uni.get('albedo_img'), vary['uv'])
    
    diff = max(np.dot(N, L), 0.0)
    base_color = albedo * (0.5 + 0.5 * diff)
    
    center_u, center_v = 0.5, 0.5
    dx, dy = u - center_u, v - center_v
    distance = math.sqrt(dx*dx + dy*dy)
    
    if distance < 0.3:  
        nebula_r = math.sin(u * 12.0 + v * 8.0) * 0.3 + 0.7
        nebula_g = math.sin(u * 15.0 - v * 10.0 + 1.57) * 0.3 + 0.7
        nebula_b = math.sin(u * 18.0 + v * 12.0 + 3.14) * 0.4 + 0.8
        
        nebula_color = np.array([nebula_r, nebula_g, nebula_b], np.float32)
        
        star_noise = math.sin(u * 80.0) * math.sin(v * 80.0)
        if star_noise > 0.95:
            star_intensity = (star_noise - 0.95) / 0.05
            star_color = np.array([1.0, 1.0, 1.0], np.float32) * star_intensity * 0.5
            nebula_color += star_color
        
        final_color = base_color * 0.7 + nebula_color * base_color * 0.3
    else:
        final_color = base_color
    
    V = uni.get('cam_pos', np.zeros(3, np.float32)) - pos
    V /= (np.linalg.norm(V) + 1e-6)
    H = L + V
    H /= (np.linalg.norm(H) + 1e-6)
    spec = max(np.dot(N, H), 0.0) ** 64.0
    final_color += spec * np.array([0.8, 0.9, 1.0], np.float32) * 0.4
    
    return np.clip(final_color, 0, 1)

def basic_shader(vary, uni):
    """Shader básico sin efectos - solo textura y iluminación simple"""
    N = vary['normal']
    N /= (np.linalg.norm(N) + 1e-6)
    L = uni.get('light_dir', np.array([0.6, 1.0, -0.5], np.float32))
    L /= (np.linalg.norm(L) + 1e-6)
    
    # Textura base - si no hay textura, usar color gris claro
    albedo = _tex2d(uni.get('albedo_img'), vary['uv'])
    if uni.get('albedo_img') is None:
        albedo = np.array([0.7, 0.7, 0.7], np.float32)
    
    # Iluminación difusa simple
    diff = max(np.dot(N, L), 0.0)
    
    # Luz ambiental más fuerte para evitar oscuridad
    ambient = 0.4
    
    # Color final con iluminación balanceada
    color = albedo * (ambient + diff * 0.6)
    
    return np.clip(color, 0, 1)

def auto_load_texture(key):
    """Carga automática de texturas con debug"""
    p = assets.get_texture(key)
    if p:
        try: 
            texture = load_texture(p)
            print(f"[DEBUG] Textura {key} cargada exitosamente: {p}")
            return texture
        except Exception as e:
            print(f"[ERROR] Error cargando textura {key}: {e}")
    else:
        print(f"[WARN] No se encontró textura para {key}")
    return None

def auto_load_normalmap(key):
    p = assets.get_normalmap(key)
    if p:
        try: return load_normal_map(p)
        except: pass
    return None


def fit_transform_floor(mesh, target_size=1.0, at=(0,0,0), yaw=0.0, pitch=0.0, roll=0.0, pre=None):
    P = np.concatenate([f['p'] for f in mesh.faces], axis=0)
    mn = P.min(axis=0); mx = P.max(axis=0)
    center = (mn+mx)*0.5
    extent = np.maximum(mx-mn, 1e-8)
    longest = float(np.max(extent))
    s = target_size / longest
    PRE = pre if pre is not None else mat4_identity()

    P0 = (P - center) @ (PRE[:3,:3].T) * s
    y_min = P0[:,1].min()
    ty = at[1] - y_min

    Ruser = rotation_y(yaw) @ rotation_x(pitch) @ rotation_z(roll)
    M = translation(at[0], ty, at[2]) @ Ruser @ PRE @ scale(s,s,s) @ translation(-center[0], -center[1], -center[2])
    return M

#cámara
FB_W, FB_H  = 1600, 900
WIN_W, WIN_H = 1280, 720

color_buf = np.zeros((FB_H, FB_W, 3), dtype=np.uint8)
z_buf     = np.full((FB_H, FB_W), np.inf, dtype=np.float32)

bg_path = os.path.join(ASSETS, "backgrounds", "fondo.bmp")
if os.path.exists(bg_path): load_background_to_buffer(color_buf, bg_path)
else: color_buf[...] = np.array([135,206,235], np.uint8)
bg_frame = color_buf.copy()

cam = Camera(eye=(0.10,1.55,6.0), target=(0.10,1.05,0.65), up=(0.0,1.0,0.0),
             fov=55.0, near=0.1, far=100.0, aspect=FB_W/float(FB_H))

light_dir = np.array([-0.25, 0.95, 0.20], np.float32)
GROUND_Y  = -1.18


SCENE = []

def create_auto_model(name, model_key, shader, size, position, rotation=(0,0,0), pre_rot=None):
    model_path = assets.get_model(model_key)
    if not model_path:
        print(f"[WARN] No se encontró modelo para {model_key}")
        return None
    print(f"[INFO] Cargando {name}: {model_path}")
    mesh = load_obj(model_path)

    albedo = auto_load_texture(model_key)
    normal = auto_load_normalmap(model_key)
    if albedo is not None: print(f"[INFO] Textura cargada para {name}")
    if normal is not None: print(f"[INFO] Normal map cargado para {name}")

    M = fit_transform_floor(mesh, target_size=size, at=position,
                            yaw=rotation[0], pitch=rotation[1], roll=rotation[2], pre=pre_rot)
    item = {'name':name,'mesh':mesh,'M':M,'shader':shader,'albedo':albedo,'normal':normal}
    SCENE.append(item); return item

# Pre-rotaciones
cat_pre     = rotation_x(-90) @ rotation_z( 90)
pika_pre    = rotation_x(-35)
girasol_pre = rotation_x(-90)
craneo_pre  = rotation_x(-90)

# Modelos con shaders
create_auto_model("cat",      "cat",      advanced_toon_shader,       2.30, (-3.20, GROUND_Y, -1.95), (50,-5,0),  cat_pre)
create_auto_model("pikachu",  "pikachu",  neon_wireframe_shader,      2.10, (-0.38, GROUND_Y, -1.22), (20,-10,0), pika_pre)
create_auto_model("eye",      "eye",      galaxy_shader,              1.20, (1.85, GROUND_Y+2.95, -1.35), (-12,0,0), None)
create_auto_model("sunflower","sunflower",metallic_shader,            3.10, (1.00, GROUND_Y-0.18, -1.05), (-6,0,0), girasol_pre)
create_auto_model("skull",    "skull",    advanced_checkerboard_shader,2.10, (2.55, GROUND_Y+0.06, -1.40), (-35,0,0), craneo_pre)

# Render
def render_scene():
    """Renderiza la escena con shaders avanzados"""
    color_buf[...] = bg_frame
    z_buf[:] = np.inf
    for it in SCENE:
        uniforms = dict(
            light_dir=light_dir,
            cam_pos=np.array(cam.eye, np.float32),
            albedo_img=it['albedo'],
            normalmap_img=it['normal'],
            ka=0.55, kd=0.85, ks=0.25, shininess=32.0
        )
        draw_mesh(it['mesh'], it['M'], cam.V, cam.P, color_buf, z_buf, uniforms, it['shader'])

def render_scene_basic():
    """Renderiza la escena con shader básico para todos los modelos"""
    print("[DEBUG] Iniciando render básico...")
    color_buf[...] = bg_frame
    z_buf[:] = np.inf
    for it in SCENE:
        print(f"[DEBUG] Renderizando {it['name']} con shader básico")
        print(f"[DEBUG] - Tiene textura: {it['albedo'] is not None}")
        if it['albedo'] is not None:
            print(f"[DEBUG] - Forma de textura: {it['albedo'].shape}")
        
        uniforms = dict(
            light_dir=light_dir,
            cam_pos=np.array(cam.eye, np.float32),
            albedo_img=it['albedo'],
            normalmap_img=it['normal'],
            ka=0.55, kd=0.85, ks=0.25, shininess=32.0
        )
        draw_mesh(it['mesh'], it['M'], cam.V, cam.P, color_buf, z_buf, uniforms, basic_shader)

def save_frame(name, folder="outputs"):
    out_abs = os.path.join(ASSETS, folder, f"{name}.bmp")
    os.makedirs(os.path.dirname(out_abs), exist_ok=True)
    save_bmp(out_abs, color_buf)
    print(f"[OK] Guardado: {out_abs}")

if __name__ == "__main__":
    print("[INFO] Renderizando escena con shaders avanzados...")
    render_scene()
    save_frame("scene_shaders_avanzados")
    print("[SUCCESS] Escena con shaders avanzados exportada")
    
    print("[INFO] Renderizando escena sin shaders...")
    render_scene_basic()
    save_frame("scene_sin_shaders")
    print("[SUCCESS] Escena sin shaders exportada")
    
    print("[SUCCESS] Ambas versiones guardadas en assets/outputs/")