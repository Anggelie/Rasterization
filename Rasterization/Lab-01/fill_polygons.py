# Universidad del Valle de Guatemala 
# Facultad de Ingeniería
# Departamento de Ciencias de la Computación
# Curso: Gráficas por Computadora
# Catedrático: Ing. Carlos Alonso 
# Anggelie Velásquez 221181
# Laboratorio 1 - Filling any Polygon

import tkinter as tk
from math import sin, cos, radians

def translate(points, dx, dy):
    return [(x + dx, y + dy) for (x, y) in points]

def rotate(points, angle_deg):
    angle = radians(angle_deg)
    return [
        (
            int(x * cos(angle) - y * sin(angle)),
            int(x * sin(angle) + y * cos(angle))
        )
        for x, y in points
    ]

def flip_polygon_vertically(points):
    ys = [y for (_, y) in points]
    y_center = (min(ys) + max(ys)) // 2
    return [(x, 2 * y_center - y) for (x, y) in points]

polygon1 = [(0, 0), (20, -20), (15, -50), (42, -35), (68, -50),
            (65, -20), (85, 0), (55, 5), (40, 30), (28, 3)]

polygon2 = [(0, 0), (-33, -49), (18, -84), (53, -33)]

polygon3 = [(0, 0), (34, -52), (59, 0)]

polygon4 = flip_polygon_vertically([
    (500, 280), (535, 262), (589, 191), (640, 156), (622, 139),
    (763, 140), (747, 155), (837, 248), (848, 282), (759, 295),
    (746, 317), (702, 317), (719, 333), (667, 333), (684, 318),
    (639, 317), (604, 247), (553, 283)
])

hole = [(755, 245), (781, 190), (808, 218), (812, 240)]

# Algoritmo de Scanline Fill. Este algoritmo lo consulté con una IA para entender cómo implementar el relleno de polígonos.
def scanline_fill(canvas, points, fill_color="#FFADAD"):
    edges = []
    n = len(points)

    for i in range(n):
        x0, y0 = points[i]
        x1, y1 = points[(i + 1) % n]
        if y0 == y1:
            continue
        if y0 > y1:
            x0, y0, x1, y1 = x1, y1, x0, y0
        edges.append({'x0': x0, 'y0': y0, 'x1': x1, 'y1': y1,
                      'inv_slope': (x1 - x0) / (y1 - y0)})

    ymin = min(p[1] for p in points)
    ymax = max(p[1] for p in points)

    for y in range(ymin, ymax):
        intersections = []
        for edge in edges:
            if edge['y0'] <= y < edge['y1']:
                x = edge['x0'] + (y - edge['y0']) * edge['inv_slope']
                intersections.append(x)

        intersections.sort()
        for i in range(0, len(intersections), 2):
            if i + 1 < len(intersections):
                x_start = int(intersections[i])
                x_end = int(intersections[i + 1])
                canvas.create_line(x_start, y, x_end, y, fill=fill_color)

def main():
    window = tk.Tk()
    window.title("Polygon Fill Lab - Anggelie Velásquez")
    canvas = tk.Canvas(window, width=900, height=400, bg="white", highlightthickness=0)
    canvas.pack()

    # Paleta de colores pastel elegidos para dar un estilo visual armónico
    pastel_colors = {
        "rosa": "#FFADAD",         # Tetera
        "durazno": "#FFD6A5",      # Estrella
        "menta": "#E4F1EE",        # Cuadrado
        "azul_cielo": "#D9EDF8",   # Triángulo
    }

    estrella = rotate(polygon1, 15)
    estrella = translate(estrella, 150, 100)
    scanline_fill(canvas, estrella, fill_color=pastel_colors["durazno"])
    canvas.create_polygon(estrella, outline="#AAAAAA", fill="", width=1)

    cuadrado = rotate(polygon2, 45)
    cuadrado = translate(cuadrado, 250, 160)
    scanline_fill(canvas, cuadrado, fill_color=pastel_colors["menta"])
    canvas.create_polygon(cuadrado, outline="#AAAAAA", fill="", width=1)

    triangulo = rotate(polygon3, -25)
    triangulo = translate(triangulo, 330, 220)
    scanline_fill(canvas, triangulo, fill_color=pastel_colors["azul_cielo"])
    canvas.create_polygon(triangulo, outline="#AAAAAA", fill="", width=1)

    scanline_fill(canvas, polygon4, fill_color=pastel_colors["rosa"])
    canvas.create_polygon(polygon4, outline="#AAAAAA", fill="", width=1)

    # Agujero en blanco (se simula sobre la tetera) – Esto también fue consultado con IA para entender cómo hacerlo visualmente
    scanline_fill(canvas, hole, fill_color="white") 
    canvas.create_polygon(hole, outline="#AAAAAA", fill="", width=1)

    window.mainloop()

if __name__ == "__main__":
    main()
