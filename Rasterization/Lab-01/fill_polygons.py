# Universidad del Valle de Guatemala
# Facultad de Ingeniería
# Departamento de Ciencias de la Computación
# Curso: Gráficas por Computadora
# Catedrático: Ing. Carlos Alonso 
# Anggelie Velásquez 221181
# Laboratorio 1 - Filling any Polygon

import tkinter as tk

# Polígonos
polygon1 = [(100, 300), (120, 280), (115, 250), (142, 265), (168, 250),
            (165, 280), (185, 300), (155, 305), (140, 330), (128, 303)]

polygon2 = [(300, 100), (267, 51), (318, 16), (353, 67)]

polygon3 = [(200, 120), (234, 68), (259, 120)]

def flip_polygon_vertically(points):
    ys = [y for (_, y) in points]
    y_center = (min(ys) + max(ys)) // 2
    return [(x, 2 * y_center - y) for (x, y) in points]

# Tetera y agujero reflejados correctamente
polygon4 = flip_polygon_vertically([
    (500, 280), (535, 262), (589, 191), (640, 156), (622, 139),
    (763, 140), (747, 155), (837, 248), (848, 282), (759, 295),
    (746, 317), (702, 317), (719, 333), (667, 333), (684, 318),
    (639, 317), (604, 247), (553, 283)
])

hole = flip_polygon_vertically([
    (730, 278), (756, 223), (783, 251), (787, 273)
])

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
    window.title("Polygon Fill Lab - Anggelie")
    canvas = tk.Canvas(window, width=900, height=400, bg="white", highlightthickness=0)
    canvas.pack()

    pastel_colors = {
        "rosa": "#FFADAD",
        "durazno": "#FFD6A5",
        "amarillo": "#FDFFB6",
        "menta": "#E4F1EE",
        "azul_cielo": "#D9EDF8",
        "lavanda": "#DEDAF4"
    }

    scanline_fill(canvas, polygon2, fill_color=pastel_colors["menta"])
    canvas.create_polygon(polygon2, outline="#AAAAAA", fill="", width=1)

    scanline_fill(canvas, polygon1, fill_color=pastel_colors["durazno"])
    canvas.create_polygon(polygon1, outline="#AAAAAA", fill="", width=1)

    scanline_fill(canvas, polygon3, fill_color=pastel_colors["azul_cielo"])
    canvas.create_polygon(polygon3, outline="#AAAAAA", fill="", width=1)

    scanline_fill(canvas, polygon4, fill_color=pastel_colors["rosa"])
    canvas.create_polygon(polygon4, outline="#AAAAAA", fill="", width=1)

    scanline_fill(canvas, hole, fill_color="white")  # Agujero en blanco
    canvas.create_polygon(hole, outline="#AAAAAA", fill="", width=1)

    window.mainloop()

if __name__ == "__main__":
    main()
