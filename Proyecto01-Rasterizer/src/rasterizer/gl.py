import numpy as np
from PIL import Image

def load_background_to_buffer(buffer: np.ndarray, path: str):
    h, w = buffer.shape[:2]
    img = Image.open(path).convert("RGB").resize((w, h), Image.BICUBIC)
    buffer[:, :, :] = np.asarray(img, dtype=np.uint8)

def save_bmp(path, rgb_buffer: np.ndarray):
    img = Image.fromarray(rgb_buffer, mode="RGB")
    img.save(path, format="BMP")
