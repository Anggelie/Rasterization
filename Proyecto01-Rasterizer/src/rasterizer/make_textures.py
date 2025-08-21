import os, numpy as np
from PIL import Image, ImageDraw

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASSETS = os.path.join(BASE, "assets")
OUTDIR = os.path.join(ASSETS, "textures")
os.makedirs(OUTDIR, exist_ok=True)

W, H = 1024, 1024

def save(img, name):
    p = os.path.join(OUTDIR, name)
    img.save(p, "PNG")
    print("[OK]", p)

def tri_tiling(palette):
    """Patrón de triángulos tipo UV + líneas negras finas"""
    img = Image.new("RGB", (W, H), (32, 32, 32))
    drw = ImageDraw.Draw(img)
    n = 16  
    cw, ch = W//n, H//n
    k = 0
    for j in range(n):
        for i in range(n):
            x0, y0 = i*cw, j*ch
            x1, y1 = x0+cw, y0+ch
            c = palette[k % len(palette)]; k += 1
            # tri 1
            drw.polygon([(x0,y0),(x1,y0),(x1,y1)], fill=c)
            # tri 2
            c2 = palette[(k+3) % len(palette)]
            drw.polygon([(x0,y0),(x0,y1),(x1,y1)], fill=c2)
            # líneas
            drw.line([(x0,y0),(x1,y0)], fill=(0,0,0), width=1)
            drw.line([(x0,y0),(x0,y1)], fill=(0,0,0), width=1)
            drw.line([(x0,y1),(x1,y1)], fill=(0,0,0), width=1)
            drw.line([(x1,y0),(x1,y1)], fill=(0,0,0), width=1)
            drw.line([(x0,y0),(x1,y1)], fill=(0,0,0), width=1)
    return img

def grid_wire(color_bg=(15,15,20), color_line=(30,160,255)):
    """Fondo oscuro con grilla fina: ideal para wireframe"""
    img = Image.new("RGB", (W, H), color_bg)
    drw = ImageDraw.Draw(img)
    step = 32
    for x in range(0, W, step):
        drw.line([(x,0),(x,H)], fill=color_line, width=1)
    for y in range(0, H, step):
        drw.line([(0,y),(W,y)], fill=color_line, width=1)
    return img

def iris_like():
    """Iris simple: anillos/segmentos para el ojo"""
    img = Image.new("RGB", (W, H), (235, 235, 235))
    drw = ImageDraw.Draw(img)
    cx, cy = W//2, H//2
    for r in range(min(cx,cy), 40, -8):
        col = (int(60+80*np.sin(r*0.1)),
               int(120+100*np.sin(r*0.07)),
               int(160+80*np.sin(r*0.09)))
        drw.ellipse([cx-r, cy-r, cx+r, cy+r], outline=(20,20,20), width=2, fill=col)
    drw.ellipse([cx-90, cy-90, cx+90, cy+90], fill=(15,15,20))
    return img

def checker(b0=(20,20,20), b1=(230,230,230), scale=16):
    img = Image.new("RGB", (W, H), b0)
    drw = ImageDraw.Draw(img)
    cw = W//scale
    for j in range(scale):
        for i in range(scale):
            if (i+j) & 1:
                x0, y0 = i*cw, j*cw
                drw.rectangle([x0, y0, x0+cw, y0+cw], fill=b1)
    return img

if __name__ == "__main__":
    cat = tri_tiling([(210,120,40),(170,90,30),(240,160,60),(120,70,30)])
    save(cat, "cat_uv.png")

    pika = grid_wire()
    save(pika, "pikachu_uv.png")

    eye = iris_like()
    save(eye, "eye_uv.png")

    sun = tri_tiling([(60,110,40),(90,150,50),(170,160,40),(210,190,60)])
    save(sun, "sunflower_uv.png")

    skull = checker()
    save(skull, "skull_uv.png")

    print("\nListo. Colocadas en assets/textures/*.png")
