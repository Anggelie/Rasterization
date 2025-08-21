import numpy as np

def GenerateBMP(filename, width, height, channels, frameBuffer):
    """
    Genera un archivo BMP a partir del frameBuffer
    """
    try:
        # Convertir frameBuffer a formato correcto
        if isinstance(frameBuffer, np.ndarray):
            if frameBuffer.dtype != np.uint8:
                frameBuffer = (frameBuffer * 255).astype(np.uint8)
            
            # Asegurar que esté en formato RGB
            if len(frameBuffer.shape) == 3 and frameBuffer.shape[2] >= 3:
                # Tomar solo los primeros 3 canales (RGB)
                img_data = frameBuffer[:, :, :3]
            else:
                print(f"Warning: frameBuffer shape {frameBuffer.shape} no es válido")
                return
            
            # Crear una imagen simple usando PIL si está disponible
            try:
                from PIL import Image
                # Convertir de RGB a BGR para BMP
                img_data_bgr = img_data[:, :, [2, 1, 0]]
                img = Image.fromarray(img_data_bgr)
                img.save(filename)
                print(f"Imagen guardada como {filename}")
            except ImportError:
                # Si PIL no está disponible, guardar como archivo raw
                with open(filename.replace('.bmp', '.raw'), 'wb') as f:
                    img_data.tobytes()
                print(f"PIL no disponible. Guardado como {filename.replace('.bmp', '.raw')}")
                
    except Exception as e:
        print(f"Error guardando imagen: {e}")