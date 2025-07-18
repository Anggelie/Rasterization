# Laboratorio 2 – OBJ Models

## Universidad del Valle de Guatemala  
**Facultad de Ingeniería**  
**Departamento de Ciencias de la Computación**  
Curso: Gráficas por Computadora  
Catedrático: Ing. Carlos Alonso  
**Nombre:** Anggelie Velásquez – 221181  
**Lab 2 – Rasterización y Visualización de Modelos OBJ**

---

## Objetivo

El objetivo de este laboratorio es desarrollar un visualizador de modelos 3D en formato `.obj`, renderizado mediante puntos, líneas y triángulos. El modelo debe poder:

- Dibujarse usando colores aleatorios para cada triángulo.
- Ser transformado mediante matrices de Traslación, Rotación y Escala.
- Centrarse y escalarse automáticamente en pantalla para una correcta visualización.
- Exportarse como archivo BMP desde cada modo (1=puntos, 2=líneas, 3=triángulos).

---

## Lo aprendido

- Rasterización manual de líneas y triángulos en 2D.
- Uso del formato OBJ y carga de modelos 3D personalizados.
- Transformaciones en espacio tridimensional.
- Generación de imágenes BMP desde buffer.
- Visualización con Pygame.

---

## 🛠️ Archivos principales

| Archivo | Descripción |
|--------|-------------|
| `Rasterizer2025.py` | Control principal de la ejecución y eventos. |
| `gl.py` | Funciones básicas de rasterización y canvas. |
| `model.py` | Contiene la clase `Model` para transformar y renderizar el modelo. |
| `OBJ.py` | Parser para cargar archivos `.obj`. |
| `shaders.py` | Shader simple para aplicar colores aleatorios. |
| `girlOBJ.obj` | Modelo 3D usado en el laboratorio (formato OBJ). |

---

## Instrucciones de uso

1. Ejecuta el archivo `Rasterizer2025.py`.
2. Usa las teclas para cambiar de modo:
   - `1`: puntos
   - `2`: líneas
   - `3`: triángulos
3. Usa `W`, `S`, `A`, `D` para mover el modelo.
4. Usa `Q`, `E` para rotarlo y `Z`, `X` para escalarlo.
5. Se genera automáticamente un archivo `output.bmp` al cambiar de modo.

---

## Ejemplos de visualización
Los podras encontrar en la carpeta Pruebas del funcionamiento

---

##  Estado final

✔ Modelo se carga centrado  
✔ Se visualiza en los tres modos  
✔ Triángulos correctamente rasterizados con color  
✔ Transformaciones aplicadas con matriz  
✔ Imagen BMP generada con éxito

---

## Créditos

> Este proyecto fue desarrollado por **Anggelie Lizeth Velásquez Asencio** como parte del curso de **Gráficas por Computadora** - Primer Ciclo 2025 – UVG.

> Algunas partes del README se hicieron con ayuda de [ChatGPT](https://chat.openai.com/) para poder organizarlo de una manera más clara.