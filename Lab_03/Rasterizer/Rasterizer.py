# Universidad del Valle de Guatemala
# Facultad de Ingeniería
# Departamento de Ciencias de la Computación
# Curso: Gráficas por Computadora
# Catedrático: Ing. Carlos Alonso
# Anggelie Velásquez 221181
# Laboratorio 3 – Cámara Manual con Textura y UV (FINAL FUNCIONAL)

import pygame
import numpy as np
import sys
import os
from pygame.locals import *

from models.OBJ import load_obj_with_mtl
from models.model import Model
from models.gl import Renderer, TRIANGLES
from models.shaders import vertexShader, fragmentShader
from models.pipeline_matrices import get_view_matrix, get_projection_matrix, get_viewport_matrix
from models.BMP_Writer import GenerateBMP

# Dimensiones de la ventana
WIDTH = 800
HEIGHT = 800

pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Lab 3 - Cámara Manual con Textura")

# Inicializar renderer
r = Renderer(screen)

# Ruta al modelo OBJ
obj_path = os.path.join("..", "girlOBJ.obj")

# Cargar datos del modelo
vertices, faces, materials, texcoords = load_obj_with_mtl(obj_path)

# Calcular centro y escala del modelo
all_vertices = np.array([v for v in vertices])
min_vals = np.min(all_vertices, axis=0)
max_vals = np.max(all_vertices, axis=0)
center = (min_vals + max_vals) / 2
size = np.max(max_vals - min_vals)
scale = 5.0 / size

# Preparar modelo
model = Model(obj_path)
model.vertexShader = vertexShader
model.fragmentShader = fragmentShader
model.vertices = []
model.texcoords = []
model.colors = []
model.texture = None

# Cargar texturas desde los materiales
material_textures = {}
for name, data in materials.items():
    if "map_Kd" in data:
        texture_path = data["map_Kd"]
        if os.path.isfile(texture_path):
            texture = pygame.image.load(texture_path)
            material_textures[name] = texture
        else:
            print(f"No se pudo cargar: {texture_path}")

# Procesar caras y texturas
for face, mat in faces:
    texture = material_textures.get(mat)
    if texture is not None:
        model.texture = texture

    if len(face) >= 3:
        for j in range(1, len(face) - 1):
            for idx in [face[0], face[j], face[j + 1]]:
                vx, vy, vz = vertices[idx[0]]
                sx = (vx - center[0]) * scale
                sy = (vy - center[1]) * scale
                sz = (vz - center[2]) * scale
                model.vertices.extend([sx, sy, sz])

                if idx[1] is not None and idx[1] < len(texcoords):
                    model.texcoords.extend(texcoords[idx[1]])
                else:
                    model.texcoords.extend([0, 0])

                model.colors.append([1, 1, 1])

model.translation = [0, 0, 0]
model.scale = [1, 1, 1]

eye = [0, 0, 8]
target = [0, 0, 0]
up = [0, 1, 0]

view = get_view_matrix(eye, target, up)
projection = get_projection_matrix(45, WIDTH / HEIGHT, 0.1, 1000)
viewport = get_viewport_matrix(0, 0, WIDTH, HEIGHT)

model.transform = viewport @ projection @ view

r.models = [model]

# Variables para rotación
angle_x = 0
angle_y = 0
mouse_down = False
last_mouse_pos = (0, 0)

clock = pygame.time.Clock()
running = True

print("Controles:")
print(" - Mantén clic izquierdo y mueve para rotar el modelo")
print(" - Usa flechas del teclado para rotar también")
print(" - S para guardar imagen como output.bmp")
print(" - ESC para salir")

while running:
    for event in pygame.event.get():
        if event.type == QUIT:
            running = False

        elif event.type == KEYDOWN:
            if event.key == K_ESCAPE:
                running = False
            elif event.key == K_LEFT:
                angle_y -= 5
            elif event.key == K_RIGHT:
                angle_y += 5
            elif event.key == K_UP:
                angle_x -= 5
            elif event.key == K_DOWN:
                angle_x += 5
            elif event.key == K_s:
                GenerateBMP("output.bmp", screen)
                print("Imagen guardada como output.bmp")

        elif event.type == MOUSEBUTTONDOWN:
            if event.button == 1:
                mouse_down = True
                last_mouse_pos = pygame.mouse.get_pos()

        elif event.type == MOUSEBUTTONUP:
            if event.button == 1:
                mouse_down = False

        elif event.type == MOUSEMOTION and mouse_down:
            x, y = pygame.mouse.get_pos()
            dx = x - last_mouse_pos[0]
            dy = y - last_mouse_pos[1]
            angle_y += dx * 0.5
            angle_x += dy * 0.5
            last_mouse_pos = (x, y)

    model.rotation = [np.radians(angle_x), np.radians(angle_y), 0]
    r.clear()
    r.draw_arrays(TRIANGLES)
    r.display()
    clock.tick(60)

pygame.quit()
