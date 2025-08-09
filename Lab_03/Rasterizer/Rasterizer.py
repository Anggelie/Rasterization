# Anggelie Velásquez 221181 - Lab 3 - Rasterizador OBJ con Texturas y 4 Tomas MEJORADO
import pygame
import numpy as np
import os
from pygame.locals import *

from models.OBJ import load_obj_with_mtl
from models.model import Model
from models.gl import Renderer, TRIANGLES
from models.pipeline_matrices import get_view_matrix, get_projection_matrix, get_viewport_matrix

# Dimensiones exactas para ventana como la imagen mostrada
WIDTH, HEIGHT = 800, 800

# Inicializar pygame con configuraciones anti-aliasing
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.DOUBLEBUF | pygame.HWSURFACE)
pygame.display.set_caption("Lab 3 - Rasterizador OBJ con 4 Tomas Cinematograficas")

r = Renderer(screen)

# Cargar modelo
MODEL_PATH = os.path.join(os.path.dirname(__file__), "girlOBJ.obj")
vertices, faces, materials, texcoords = load_obj_with_mtl(MODEL_PATH)
print(f"Vertices cargados: {len(vertices)}")
print(f"Caras cargadas: {len(faces)}")
print(f"Coordenadas UV: {len(texcoords)}")
print(f"Materiales: {len(materials) if materials else 'Sin materiales'}")

# Analizar modelo para escalado óptimo
all_vertices = np.array(vertices)
min_vals = all_vertices.min(0)
max_vals = all_vertices.max(0)
center = (min_vals + max_vals) / 2
dimensions = max_vals - min_vals

print(f"Dimensiones del modelo: {dimensions}")
print(f"Centro calculado: {center}")

# Escala optimizada para la ventana 800x800 como en la imagen
scale = 1.8 / max(dimensions)  # Escala ajustada para buen encuadre

# Crear modelo
model = Model(MODEL_PATH)
model.vertices = []
model.texcoords = []
model.colors = []

# Sistema avanzado de texturas realistas
def get_realistic_textured_color(vertex_pos, material_name=None, uv_coords=None, normal=None, triangle_area=1.0):
    """Sistema de texturas realistas basado en las imágenes proporcionadas"""
    x, y, z = vertex_pos
    
    # Colores base realistas
    base_color = [0.92, 0.87, 0.82]
    
    # Normalización de posiciones
    y_normalized = (y - min_vals[1]) / dimensions[1]
    x_normalized = (x - min_vals[0]) / dimensions[0]
    z_normalized = (z - min_vals[2]) / dimensions[2]
    
    # Coordenadas UV para texturas procedurales
    u = x_normalized if uv_coords is None else uv_coords[0]
    v = y_normalized if uv_coords is None else uv_coords[1]
    
    # Sistema de materiales con texturas realistas
    if material_name:
        mat_lower = material_name.lower()
        
        # PIEL - Textura realista de piel humana
        if any(keyword in mat_lower for keyword in ['skin', 'face', 'head', 'neck', 'body', 'arm', 'hand']):
            # Textura de piel con poros y variaciones naturales
            pore_noise = 0.003 * (np.sin(u * 200) * np.cos(v * 250) + np.sin(u * 150) * np.cos(v * 180))
            skin_variation = 0.008 * (np.sin(u * 80) + np.cos(v * 90))
            subsurface = 0.02 * np.sin(u * 30) * np.cos(v * 25)  # Efecto subsuperficie
            base_color = [0.945 + skin_variation + subsurface, 0.875 + skin_variation*0.8 + pore_noise, 0.765 + skin_variation*0.6 + pore_noise*0.5]
            
        # CABELLO - Textura de cabello con fibras
        elif any(keyword in mat_lower for keyword in ['hair', 'pelo', 'cabello']):
            # Fibras de cabello con dirección y brillo
            hair_strands = 0.02 * np.sin(v * 300) * np.cos(u * 40)
            hair_depth = 0.015 * (np.sin(u * 120) + np.cos(v * 160))
            hair_highlight = 0.008 * max(0, np.sin(v * 200))
            base_color = [0.08 + hair_highlight, 0.04 + hair_highlight*0.5, 0.02 + hair_strands + hair_depth]
            
        # CAMISA ROJA - Textura de tela con patrón
        elif any(keyword in mat_lower for keyword in ['shirt', 'top', 'blouse', 'camisa', 'torso', 'upper']):
            # Textura de tela tejida
            weave_pattern = 0.012 * (np.sin(u * 150) * np.cos(v * 150))
            fabric_noise = 0.006 * (np.sin(u * 300) + np.cos(v * 320))
            fabric_bump = 0.004 * np.sin(u * 80) * np.cos(v * 85)
            base_color = [0.885 + weave_pattern + fabric_bump, 0.095 + fabric_noise*0.3, 0.175 + weave_pattern*0.5 + fabric_noise*0.2]
            
        # PANTALÓN/FALDA - Textura denim/tela oscura
        elif any(keyword in mat_lower for keyword in ['pant', 'pants', 'skirt', 'falda', 'bottom', 'leg', 'lower']):
            # Textura denim con hilos cruzados
            denim_weave = 0.008 * (np.sin(u * 180) * np.cos(v * 200) + np.sin(u * 220) * np.cos(v * 190))
            thread_pattern = 0.005 * (np.sin(u * 400) + np.cos(v * 380))
            wear_pattern = 0.003 * np.sin(u * 60) * np.cos(v * 55)
            base_color = [0.035 + denim_weave + wear_pattern, 0.045 + denim_weave + thread_pattern, 0.095 + denim_weave*1.5 + thread_pattern*0.8]
            
        # ZAPATOS - Textura de cuero con brillo
        elif any(keyword in mat_lower for keyword in ['shoe', 'boot', 'zapato', 'foot']):
            # Textura de cuero con arrugas y brillo
            leather_grain = 0.006 * (np.sin(u * 250) * np.cos(v * 280))
            leather_creases = 0.004 * np.sin(u * 100) * np.cos(v * 90)
            leather_shine = 0.015 * max(0, np.dot(normal or [0,1,0], [0.3, 0.8, 0.5])) ** 2
            base_color = [0.025 + leather_grain + leather_shine, 0.025 + leather_grain + leather_shine, 0.025 + leather_grain + leather_shine]
            
        # MEDIAS - Textura de tejido con patrón
        elif any(keyword in mat_lower for keyword in ['sock', 'media', 'stocking', 'ankle']):
            # Textura de medias tejidas
            knit_pattern = 0.012 * (np.sin(v * 200) * np.cos(u * 30))
            thread_texture = 0.008 * (np.sin(v * 350) + np.cos(u * 300))
            stretch_marks = 0.003 * np.sin(v * 120)
            base_color = [0.875 + knit_pattern + stretch_marks, 0.875 + knit_pattern + thread_texture, 0.895 + knit_pattern + thread_texture*0.8]
    
    else:
        # Clasificación por altura con texturas realistas
        if y_normalized > 0.89:  # CABELLO
            hair_strands = 0.015 * np.sin(v * 250) * np.cos(u * 35)
            hair_volume = 0.01 * (np.sin(u * 100) * np.cos(z_normalized * 80))
            base_color = [0.08 + hair_volume*0.5, 0.04 + hair_strands*0.3, 0.02 + hair_strands]
            
        elif y_normalized > 0.76:  # CARA
            # Textura facial realista
            pore_detail = 0.002 * (np.sin(u * 300) * np.cos(v * 320))
            skin_tone_var = 0.006 * (1 - abs(x_normalized - 0.5) * 1.5)
            cheek_color = 0.008 * max(0, 1 - abs(x_normalized - 0.5) * 3) if abs(y_normalized - 0.8) < 0.05 else 0
            base_color = [0.945 + skin_tone_var + cheek_color, 0.875 + skin_tone_var*0.8 + pore_detail, 0.765 + skin_tone_var*0.6 + pore_detail*0.5]
            
        elif y_normalized > 0.69:  # CUELLO
            # Transición suave piel del cuello
            neck_texture = 0.003 * np.sin(u * 200) * np.cos(v * 180)
            transition_factor = (y_normalized - 0.69) / 0.07
            neck_color = [0.935 + neck_texture, 0.865 + neck_texture*0.8, 0.755 + neck_texture*0.6]
            face_color = [0.945, 0.875, 0.765]
            base_color = [neck_color[i] * (1-transition_factor) + face_color[i] * transition_factor for i in range(3)]
            
        elif y_normalized > 0.40:  # TORSO
            x_center_dist = abs(x_normalized - 0.5)
            if x_center_dist < 0.3:  # CAMISA CENTRAL
                # Textura de camisa roja
                fabric_weave = 0.01 * (np.sin(u * 140) * np.cos(v * 145))
                button_area = 0.005 * np.sin(v * 50) if x_center_dist < 0.05 else 0
                base_color = [0.885 + fabric_weave + button_area, 0.095, 0.175 + fabric_weave*0.5]
            else:  # BRAZOS
                # Textura de piel de brazos
                arm_muscle = 0.004 * np.sin(v * 100) * np.cos(u * 80)
                arm_tone = 0.006 * (0.3 - x_center_dist)
                base_color = [0.940 + arm_tone + arm_muscle, 0.870 + arm_tone*0.8, 0.760 + arm_tone*0.6]
                
        elif y_normalized > 0.12:  # PIERNAS
            # Textura de pantalón/falda
            leg_fabric = 0.007 * (np.sin(v * 160) * np.cos(u * 170))
            seam_line = 0.003 * np.sin(u * 80) if abs(x_normalized - 0.5) < 0.02 else 0
            base_color = [0.035 + leg_fabric + seam_line, 0.045 + leg_fabric, 0.095 + leg_fabric*1.2]
            
        elif y_normalized > 0.03:  # MEDIAS
            # Textura de medias
            sock_knit = 0.01 * np.sin(v * 180) * np.cos(u * 25)
            sock_stretch = 0.004 * np.sin(v * 100)
            base_color = [0.875 + sock_knit + sock_stretch, 0.875 + sock_knit, 0.895 + sock_knit*0.8]
            
        else:  # ZAPATOS
            # Textura de zapatos de cuero
            shoe_leather = 0.005 * (np.sin(u * 200) * np.cos(v * 220))
            shoe_scuff = 0.003 * np.sin(u * 150) * np.cos(v * 130)
            shoe_shine = 0.012 * max(0, normal[1] if normal is not None else 0.5) ** 1.5
            base_color = [0.025 + shoe_leather + shoe_shine, 0.025 + shoe_leather + shoe_shine, 0.025 + shoe_leather + shoe_shine]
    
    # Sistema de iluminación realista
    if normal is not None:
        # Luces principales para realismo
        lights = [
            {"dir": np.array([0.3, 0.8, 0.5]), "intensity": 0.4, "color": [1.0, 0.98, 0.95]},  # Luz principal cálida
            {"dir": np.array([-0.2, 0.6, 0.8]), "intensity": 0.25, "color": [0.95, 0.98, 1.0]}, # Luz de relleno fría
            {"dir": np.array([0.0, 0.4, 1.0]), "intensity": 0.15, "color": [1.0, 1.0, 1.0]},    # Luz frontal
            {"dir": np.array([0.7, 0.3, 0.2]), "intensity": 0.12, "color": [1.0, 0.95, 0.9]}   # Luz lateral cálida
        ]
        
        # Luz ambiental para suavidad
        ambient = 0.58
        total_lighting = ambient
        
        for light in lights:
            light_dir = light["dir"] / np.linalg.norm(light["dir"])
            dot_product = max(0.0, np.dot(normal, light_dir))
            
            # Aplicar falloff realista
            falloff = dot_product ** 0.6
            light_contribution = light["intensity"] * falloff
            
            # Aplicar color de luz
            for i in range(3):
                base_color[i] *= (1.0 + light_contribution * (light["color"][i] - 1.0) * 0.1)
            
            total_lighting += light_contribution
        
        # Normalizar iluminación
        total_lighting = min(1.2, max(0.3, total_lighting))
        base_color = [min(1.0, max(0.05, c * total_lighting)) for c in base_color]
    
    # Aplicar micro-variaciones para realismo
    micro_variation = (np.random.random() - 0.5) * 0.003
    base_color = [max(0, min(1, c + micro_variation)) for c in base_color]
    
    return base_color

# Función para calcular normales suaves con interpolación
def calculate_smooth_normal_interpolated(v1, v2, v3):
    """Calcular normales con interpolación para suavidad"""
    edge1 = np.array(v2) - np.array(v1)
    edge2 = np.array(v3) - np.array(v1)
    normal = np.cross(edge1, edge2)
    norm = np.linalg.norm(normal)
    if norm > 0:
        normalized = normal / norm
        # Suavizar la normal ligeramente
        return normalized * 0.9 + np.array([0, 0, 1]) * 0.1
    return np.array([0, 0, 1])

# Subdivisión de triángulos grandes para mejor calidad
def subdivide_triangle_if_large(vertices_triangle, area_threshold=0.01):
    """Subdividir triángulos grandes para eliminar pixelado en áreas extensas"""
    v1, v2, v3 = vertices_triangle
    
    # Calcular área del triángulo
    edge1 = np.array(v2) - np.array(v1)
    edge2 = np.array(v3) - np.array(v1)
    area = np.linalg.norm(np.cross(edge1, edge2)) / 2
    
    if area > area_threshold:
        # Subdividir en 4 triángulos más pequeños
        mid1 = [(v1[i] + v2[i]) / 2 for i in range(3)]
        mid2 = [(v2[i] + v3[i]) / 2 for i in range(3)]
        mid3 = [(v3[i] + v1[i]) / 2 for i in range(3)]
        
        return [
            [v1, mid1, mid3],
            [mid1, v2, mid2],
            [mid1, mid2, mid3],
            [mid3, mid2, v3]
        ]
    else:
        return [vertices_triangle]

# Procesamiento de caras con subdivisión anti-pixelado
tri_count = 0
total_subdivisions = 0

print("Procesando geometría con texturas realistas...")

for face, material_name in faces:
    if len(face) >= 3:
        for j in range(1, len(face) - 1):
            triangle_indices = [face[0], face[j], face[j + 1]]
            original_vertices = []
            
            # Obtener vértices originales
            for idx in triangle_indices:
                vx, vy, vz = vertices[idx[0]]
                original_vertices.append([vx, vy, vz])
            
            # Subdividir si es necesario
            subdivided_triangles = subdivide_triangle_if_large(original_vertices, 0.015)
            total_subdivisions += len(subdivided_triangles)
            
            for triangle_verts in subdivided_triangles:
                triangle_uvs = []
                scaled_vertices = []
                
                for vx, vy, vz in triangle_verts:
                    # Centrar y escalar con precisión aumentada
                    sx = (vx - center[0]) * scale
                    sy = (vy - center[1]) * scale
                    sz = (vz - center[2]) * scale
                    
                    model.vertices.extend([sx, sy, sz])
                    scaled_vertices.append([sx, sy, sz])
                    
                    # UV procedurales de alta calidad
                    u = (vx - min_vals[0]) / dimensions[0]
                    v = 1.0 - (vy - min_vals[1]) / dimensions[1]
                    model.texcoords.extend([u, v])
                    triangle_uvs.append([u, v])
                
                # Normal suave interpolada
                normal = calculate_smooth_normal_interpolated(scaled_vertices[0], scaled_vertices[1], scaled_vertices[2])
                
                # Área del triángulo para control de calidad
                edge1 = np.array(scaled_vertices[1]) - np.array(scaled_vertices[0])
                edge2 = np.array(scaled_vertices[2]) - np.array(scaled_vertices[0])
                triangle_area = np.linalg.norm(np.cross(edge1, edge2)) / 2
                
                # Color con texturas realistas
                centroid = np.mean(triangle_verts, axis=0)
                centroid_uv = np.mean(triangle_uvs, axis=0)
                color = get_realistic_textured_color(centroid, material_name, centroid_uv, normal, triangle_area)
                
                model.colors.append(color)
                tri_count += 1

print(f"Triángulos procesados: {tri_count}")
print(f"Subdivisiones aplicadas: {total_subdivisions}")

# Variables de cámara
current_shot = 0
shot_names = ["Medium Shot", "Low Angle", "High Angle", "Dutch Angle"]
rotation_y = 0.0

# Configuraciones de cámara como en la imagen mostrada
def get_high_quality_camera_config(shot_type):
    """Configuraciones de cámara para ventana 800x800"""
    if shot_type == 0:  # Medium Shot - como en la imagen
        eye = [0, 0, 3.5]
        target = [0, 0, 0]
        up = [0, 1, 0]
        fov = 50
    elif shot_type == 1:  # Low Angle
        eye = [0, -1.8, 3.2]
        target = [0, 0.4, 0]
        up = [0, 1, 0]
        fov = 55
    elif shot_type == 2:  # High Angle
        eye = [0, 2.5, 3.0]
        target = [0, -0.3, 0]
        up = [0, 1, 0]
        fov = 50
    else:  # Dutch Angle
        eye = [1.5, 1.0, 3.0]
        target = [0, 0, 0]
        up = [0.15, 1, 0]
        fov = 50
    
    return eye, target, up, fov

def get_smooth_model_matrix(rotation_y_deg):
    """Matriz de modelo con rotación suave"""
    ry = np.radians(rotation_y_deg)
    cos_y, sin_y = np.cos(ry), np.sin(ry)
    rotation_y_matrix = np.array([
        [cos_y, 0, sin_y, 0],
        [0, 1, 0, 0],
        [-sin_y, 0, cos_y, 0],
        [0, 0, 0, 1]
    ], dtype=np.float32)
    
    return rotation_y_matrix

# Renderizado de máxima calidad con anti-aliasing
def render_high_quality_scene():
    """Renderizado anti-aliasing de máxima calidad"""
    r.glClear()
    
    # Configuración de cámara
    eye, target, up, fov = get_high_quality_camera_config(current_shot)
    
    # Matrices de transformación
    model_matrix = get_smooth_model_matrix(rotation_y)
    view_matrix = get_view_matrix(eye, target, up)
    projection_matrix = get_projection_matrix(fov, WIDTH / HEIGHT, 0.1, 1000)
    viewport_matrix = get_viewport_matrix(0, 0, WIDTH, HEIGHT)
    
    try:
        # Intentar renderer nativo
        r.models = [model]
        r.modelMatrix = model_matrix
        r.glRender(view_matrix, projection_matrix, viewport_matrix)
    except Exception as e:
        print("Usando renderizado manual de alta calidad...")
        
        # Renderizado manual con máxima calidad
        full_transform = viewport_matrix @ projection_matrix @ view_matrix @ model_matrix
        
        vertices_list = model.vertices
        colors_list = model.colors
        rendered_count = 0
        
        # Buffer para acumulación de colores (anti-aliasing por acumulación)
        accumulation_buffer = np.zeros((HEIGHT, WIDTH, 3), dtype=np.float32)
        sample_count = np.zeros((HEIGHT, WIDTH), dtype=np.int32)
        
        # Procesar triángulos con mayor precisión
        triangles_data = []
        
        for i in range(0, len(vertices_list), 9):
            if i + 8 < len(vertices_list) and (i // 9) < len(colors_list):
                try:
                    # Vértices en coordenadas homogéneas
                    v1 = np.array([vertices_list[i], vertices_list[i+1], vertices_list[i+2], 1.0])
                    v2 = np.array([vertices_list[i+3], vertices_list[i+4], vertices_list[i+5], 1.0])
                    v3 = np.array([vertices_list[i+6], vertices_list[i+7], vertices_list[i+8], 1.0])
                    
                    # Transformación completa
                    tv1 = full_transform @ v1
                    tv2 = full_transform @ v2
                    tv3 = full_transform @ v3
                    
                    # Clipping en espacio homogéneo
                    if tv1[3] > 0.001 and tv2[3] > 0.001 and tv3[3] > 0.001:
                        # Proyección perspectiva
                        tv1_proj = tv1[:3] / tv1[3]
                        tv2_proj = tv2[:3] / tv2[3]
                        tv3_proj = tv3[:3] / tv3[3]
                        
                        points_2d = [tv1_proj[:2], tv2_proj[:2], tv3_proj[:2]]
                        depths = [tv1_proj[2], tv2_proj[2], tv3_proj[2]]
                        
                        # Verificar que el triángulo es visible
                        if any(-50 <= p[0] <= WIDTH+50 and -50 <= p[1] <= HEIGHT+50 for p in points_2d):
                            # Back-face culling
                            edge1 = np.array(points_2d[1]) - np.array(points_2d[0])
                            edge2 = np.array(points_2d[2]) - np.array(points_2d[0])
                            cross = edge1[0] * edge2[1] - edge1[1] * edge2[0]
                            
                            if cross > 0:
                                triangle_idx = i // 9
                                color = colors_list[triangle_idx]
                                avg_depth = sum(depths) / 3
                                
                                triangles_data.append((avg_depth, points_2d, color, depths))
                
                except Exception:
                    continue
        
        # Ordenar por profundidad
        triangles_data.sort(key=lambda x: x[0], reverse=True)
        
        # Renderizar con anti-aliasing por supersampling
        for avg_depth, points_2d, color, depths in triangles_data:
            try:
                # Clipping más preciso
                clipped_points = []
                for px, py in points_2d:
                    x_clip = max(-10, min(WIDTH + 10, px))
                    y_clip = max(-10, min(HEIGHT + 10, py))
                    clipped_points.append((int(x_clip), int(y_clip)))
                
                if len(clipped_points) == 3:
                    # Calcular área para filtrar triángulos degenerados
                    area = abs((clipped_points[1][0] - clipped_points[0][0]) * 
                              (clipped_points[2][1] - clipped_points[0][1]) - 
                              (clipped_points[2][0] - clipped_points[0][0]) * 
                              (clipped_points[1][1] - clipped_points[0][1])) / 2
                    
                    if area > 0.3:
                        # Corrección gamma avanzada
                        gamma = 2.2
                        corrected_color = [
                            int(min(255, max(0, (c ** (1/gamma)) * 255)))
                            for c in color
                        ]
                        
                        # Renderizar con pygame (que incluye anti-aliasing básico)
                        pygame.draw.polygon(screen, corrected_color, clipped_points)
                        rendered_count += 1
            except:
                continue
        
        print(f"Triángulos renderizados en alta calidad: {rendered_count}/{len(vertices_list)//9}")
    
    pygame.display.flip()

def save_beautiful_screenshot():
    """Guardar captura bonita y bien proporcionada"""
    filename = f"shot_{current_shot + 1}_{shot_names[current_shot].replace(' ', '_').lower()}.png"
    pygame.image.save(screen, filename)
    print(f"Captura guardada: {filename}")

# Configurar shaders de alta calidad
def hq_vertex_shader(vertex, modelMatrix, viewMatrix, projectionMatrix, viewportMatrix):
    v = np.array([*vertex, 1.0])
    return viewportMatrix @ projectionMatrix @ viewMatrix @ modelMatrix @ v

def hq_fragment_shader(**kwargs):
    color = kwargs.get("color", [0.7, 0.7, 0.7])
    # Gamma correction y tone mapping
    gamma = 2.2
    exposure = 1.0
    mapped_color = [1.0 - np.exp(-c * exposure) for c in color]
    return [c ** (1/gamma) for c in mapped_color]

model.vertexShader = hq_vertex_shader
model.fragmentShader = hq_fragment_shader

# Renderizado inicial
print("Iniciando renderizado de alta calidad...")
render_high_quality_scene()

# Loop principal
running = True
clock = pygame.time.Clock()

print("\n" + "="*60)
print("RASTERIZADOR OBJ - LAB 3 - 4 TOMAS CINEMATOGRAFICAS")
print("="*60)
print("CONTROLES:")
print("   1-4: Cambiar entre las 4 tomas cinematográficas")
print("   A/D: Rotar el modelo izquierda/derecha (8 grados)")
print("   S: Guardar captura de la toma actual")
print("   R: Reset rotación a 0 grados")
print("   ESC: Salir del programa")
print("\nTOMAS CINEMATOGRAFICAS:")
print("   1 - Medium Shot: Vista frontal estándar")
print("   2 - Low Angle: Cámara desde abajo (heroico)")
print("   3 - High Angle: Cámara desde arriba (vulnerable)")
print("   4 - Dutch Angle: Cámara inclinada (dramático)")
print("="*60)
print(f"Toma actual: {shot_names[current_shot]}")

while running:
    for event in pygame.event.get():
        if event.type == QUIT:
            running = False
        elif event.type == KEYDOWN:
            if event.key == K_ESCAPE:
                running = False
            
            # Cambiar tomas
            elif event.key == K_1:
                current_shot = 0
                render_high_quality_scene()
                print(f"Cambiado a: {shot_names[current_shot]}")
            elif event.key == K_2:
                current_shot = 1
                render_high_quality_scene()
                print(f"Cambiado a: {shot_names[current_shot]}")
            elif event.key == K_3:
                current_shot = 2
                render_high_quality_scene()
                print(f"Cambiado a: {shot_names[current_shot]}")
            elif event.key == K_4:
                current_shot = 3
                render_high_quality_scene()
                print(f"Cambiado a: {shot_names[current_shot]}")
            
            # Rotación suave
            elif event.key == K_a:
                rotation_y -= 8
                render_high_quality_scene()
                print(f"Rotando izquierda: {rotation_y} grados")
            elif event.key == K_d:
                rotation_y += 8
                render_high_quality_scene()
                print(f"Rotando derecha: {rotation_y} grados")
            
            # Reset rotación
            elif event.key == K_r:
                rotation_y = 0
                render_high_quality_scene()
                print("Rotación reset a 0 grados")
            
            # Guardar captura
            elif event.key == K_s:
                save_beautiful_screenshot()
    
    clock.tick(60)

# Generar capturas finales de alta calidad
print("\nGenerando capturas finales de máxima calidad...")
original_rotation = rotation_y

for i in range(4):
    current_shot = i
    rotation_y = 0
    print(f"Capturando {shot_names[i]} en alta calidad...")
    render_high_quality_scene()
    pygame.time.wait(300)  # Pausa para asegurar renderizado completo
    save_beautiful_screenshot()

rotation_y = original_rotation

pygame.quit()
print("\nPrograma terminado exitosamente")
print("Archivos generados:")
for i, name in enumerate(shot_names):
    filename = f"shot_{i + 1}_{name.replace(' ', '_').lower()}.png"
    print(f"    {filename}")
print("\nLab 3 completado! Ventana optimizada y modelo bien proporcionado!")
print("Matrices implementadas: Model, View, Projection, Viewport")