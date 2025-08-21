# Script de diagnóstico para verificar los métodos disponibles en Renderer
import pygame
import sys
import os

# Agregar el path de los modelos si es necesario
sys.path.append(os.path.join(os.path.dirname(__file__), "models"))

try:
    from models.gl import Renderer, TRIANGLES
    
    # Inicializar pygame
    pygame.init()
    screen = pygame.display.set_mode((800, 800))
    
    # Crear instancia del renderer
    r = Renderer(screen)
    
    print("🔍 Métodos disponibles en Renderer:")
    methods = [method for method in dir(r) if not method.startswith('_')]
    for method in sorted(methods):
        print(f"  - {method}")
    
    print("\n🔍 Verificando métodos de renderizado específicos:")
    render_methods = ['draw_arrays', 'drawArrays', 'render', 'draw', 'render_triangles']
    for method in render_methods:
        if hasattr(r, method):
            print(f"  ✅ {method} - DISPONIBLE")
        else:
            print(f"  ❌ {method} - NO DISPONIBLE")
    
    pygame.quit()
    
except ImportError as e:
    print(f"❌ Error importando módulos: {e}")
    print("Verifica que los archivos en la carpeta 'models' estén correctos")
except Exception as e:
    print(f"❌ Error: {e}")