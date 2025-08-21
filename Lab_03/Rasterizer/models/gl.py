import pygame
import numpy as np
from pygame import surfarray
from pygame import image

POINTS = 0
LINES = 1
TRIANGLES = 2

class Renderer:
    def __init__(self, screen):
        self.screen = screen
        self.width, self.height = self.screen.get_size()
        self.clear_color = (0, 0, 0)
        self.vertexShader = None
        self.fragmentShader = None
        self.models = []
        self.primitiveType = TRIANGLES
        self.frameBuffer = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        self.zBuffer = np.full((self.height, self.width), np.inf)
        self.active_texture = None

    def glClear(self):
        self.frameBuffer[:] = np.array(self.clear_color) * 255
        self.zBuffer[:] = np.inf
        self.screen.fill(self.clear_color)

    def glColor(self, color):
        self.color = color

    def load_texture(self, filename):
        try:
            texture = pygame.image.load(filename)
            self.active_texture = pygame.surfarray.array3d(texture)
        except:
            print(f"Error loading texture {filename}")

    def draw_point(self, x, y, color):
        if 0 <= x < self.width and 0 <= y < self.height:
            self.frameBuffer[y, x] = np.array(color) * 255

    def transform_vertex(self, vertex, modelMatrix, viewMatrix, projectionMatrix, viewportMatrix):
        try:
            # Crear vector homogéneo
            v = np.array([vertex[0], vertex[1], vertex[2], 1.0])
            
            # Aplicar transformaciones en orden
            v = modelMatrix @ v
            v = viewMatrix @ v
            v = projectionMatrix @ v
            
            # Dividir por w para perspectiva (clip space -> NDC)
            if abs(v[3]) > 1e-8:  # Evitar división por cero
                v = v / v[3]
            
            # Aplicar transformación de viewport
            screen_pos = viewportMatrix @ v
            
            return screen_pos[:3]
        except Exception as e:
            print(f"Error en transform_vertex: {e}")
            return np.array([0, 0, 0])

    def glRender(self, viewMatrix, projectionMatrix, viewportMatrix):
        try:
            for model in self.models:
                if model.vertexShader:
                    self.vertexShader = model.vertexShader
                if model.fragmentShader:
                    self.fragmentShader = model.fragmentShader

                modelMatrix = model.GetModelMatrix()

                vertices = np.array(model.vertices).reshape(-1, 3)
                
                for i in range(0, len(vertices), 3):
                    if i + 2 < len(vertices):
                        v0 = self.transform_vertex(vertices[i], modelMatrix, viewMatrix, projectionMatrix, viewportMatrix)
                        v1 = self.transform_vertex(vertices[i+1], modelMatrix, viewMatrix, projectionMatrix, viewportMatrix)
                        v2 = self.transform_vertex(vertices[i+2], modelMatrix, viewMatrix, projectionMatrix, viewportMatrix)

                        color_idx = i // 3
                        if color_idx < len(model.colors):
                            color = model.colors[color_idx]
                        else:
                            color = [1, 1, 1]  # Color blanco por defecto

                        if self.primitiveType == POINTS:
                            self.draw_point(int(v0[0]), int(v0[1]), color)
                            self.draw_point(int(v1[0]), int(v1[1]), color)
                            self.draw_point(int(v2[0]), int(v2[1]), color)
                        elif self.primitiveType == LINES:
                            self.draw_line(v0, v1, color)
                            self.draw_line(v1, v2, color)
                            self.draw_line(v2, v0, color)
                        elif self.primitiveType == TRIANGLES:
                            self.draw_triangle(v0, v1, v2, color)

            # No actualizar la pantalla aquí - se hace en el loop principal
        except Exception as e:
            print(f"Error en glRender: {e}")
            import traceback
            traceback.print_exc()

    def draw_line(self, v0, v1, color):
        x0, y0 = int(v0[0]), int(v0[1])
        x1, y1 = int(v1[0]), int(v1[1])
        pygame.draw.line(self.screen, np.array(color) * 255, (x0, y0), (x1, y1))

    def draw_triangle(self, v0, v1, v2, color):
        points = [(int(v0[0]), int(v0[1])),
                  (int(v1[0]), int(v1[1])),
                  (int(v2[0]), int(v2[1]))]
        pygame.draw.polygon(self.screen, np.array(color) * 255, points)