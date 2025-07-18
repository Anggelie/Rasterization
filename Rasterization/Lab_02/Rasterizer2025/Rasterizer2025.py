# Universidad del Valle de Guatemala
# Facultad de Ingeniería
# Departamento de Ciencias de la Computación
# Curso: Gráficas por Computadora
# Catedrático: Ing. Carlos Alonso
# Anggelie Velásquez 221181
# Laboratorio - Visualización de Modelos OBJ

import pygame
from gl import *
from BMP_Writer import GenerateBMP
from model import Model
from shaders import *
from OBJ import load_obj
import random
import numpy as np

# Tamaño de ventana
width = 600
height = 600

# Inicialización de pantalla
pygame.display.set_caption("Visualización de modelo OBJ")
screen = pygame.display.set_mode((width, height), pygame.SCALED)
clock = pygame.time.Clock()

rend = Renderer(screen)

# Cargar modelo OBJ
obj_path = r'c:\Users\angge\Downloads\Rasterizer2025\Rasterizer2025\Rasterizer2025\Fig\girlOBJ.obj'
vertices, faces = load_obj(obj_path)

# Centrado y escalado automático
vertices_np = np.array(vertices)
min_vals = vertices_np.min(axis=0)
max_vals = vertices_np.max(axis=0)
center = (min_vals + max_vals) / 2
scale = 0.8 * min(width, height) / (max_vals - min_vals).max()

scaled_vertices = []
for x, y, z in vertices:
    sx = (x - center[0]) * scale + width // 2
    sy = (y - center[1]) * scale + height // 2
    sz = (z - center[2]) * scale
    scaled_vertices.append((sx, sy, sz))

# Buffer de triángulos - CORREGIDO (lo investigue con ayuda de chat porque no estaba funcionando)
buffer = []
colors = []

# Procesar cada cara individualmente
for face in faces:
    if len(face) == 3:
        tri_color = [random.random(), random.random(), random.random()]
        for vertex_idx in face:
            if vertex_idx < len(scaled_vertices):
                buffer.extend(scaled_vertices[vertex_idx])
        colors.append(tri_color)
    elif len(face) > 3:
        for i in range(1, len(face) - 1):
            tri_color = [random.random(), random.random(), random.random()]
            if (face[0] < len(scaled_vertices) and 
                face[i] < len(scaled_vertices) and 
                face[i + 1] < len(scaled_vertices)):
                buffer.extend(scaled_vertices[face[0]])
                buffer.extend(scaled_vertices[face[i]])
                buffer.extend(scaled_vertices[face[i + 1]])
                colors.append(tri_color)

print(f"Buffer length: {len(buffer)}")
print(f"Colors length: {len(colors)}")
print(f"Triangles: {len(buffer) // 9}")

# Crear modelo
triangleModel = Model()
triangleModel.vertices = buffer
triangleModel.colors = colors
triangleModel.vertexShader = vertexShader
triangleModel.fragmentShader = fragmentShader

rend.models.append(triangleModel)

# Loop principal
isRunning = True
while isRunning:
    deltaTime = clock.tick(60) / 1000.0

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            isRunning = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_1:
                rend.primitiveType = POINTS
            elif event.key == pygame.K_2:
                rend.primitiveType = LINES
            elif event.key == pygame.K_3:
                rend.primitiveType = TRIANGLES

    # Render
    rend.glClear()
    rend.glRender()
    pygame.display.flip()

# Guardar imagen
GenerateBMP("output.bmp", width, height, 3, rend.frameBuffer)
pygame.quit()