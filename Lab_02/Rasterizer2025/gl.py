POINTS = 0
LINES = 1
TRIANGLES = 2

class Renderer:
    def __init__(self, screen):
        self.screen = screen
        _, _, self.width, self.height = self.screen.get_rect()
        self.glColor(1, 1, 1)
        self.glClearColor(0, 0, 0)
        self.glClear()
        self.primitiveType = TRIANGLES
        self.models = []
        self.activeModelMatrix = None
        self.activeVertexShader = None

    def glClearColor(self, r, g, b):
        self.clearColor = [min(1, max(0, c)) for c in (r, g, b)]

    def glColor(self, r, g, b):
        self.currColor = [min(1, max(0, c)) for c in (r, g, b)]

    def glClear(self):
        color = [int(i * 255) for i in self.clearColor]
        self.screen.fill(color)
        self.frameBuffer = [[color for y in range(self.height)] for x in range(self.width)]

    def glPoint(self, x, y, color=None):
        x = round(x)
        y = round(y)
        if 0 <= x < self.width and 0 <= y < self.height:
            c = [int(i * 255) for i in (color or self.currColor)]
            self.screen.set_at((x, self.height - 1 - y), c)
            self.frameBuffer[x][y] = c

    def glLine(self, p0, p1, color=None):
        # Asegurar que tenemos coordenadas válidas
        if len(p0) < 2 or len(p1) < 2:
            return
        
        x0, y0 = int(round(p0[0])), int(round(p0[1]))
        x1, y1 = int(round(p1[0])), int(round(p1[1]))
        
        # Algoritmo de Bresenham mejorado
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        
        # Determinar la dirección
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        
        err = dx - dy
        
        x, y = x0, y0
        
        while True:
            # Dibujar el punto actual
            self.glPoint(x, y, color)
            
            # Si llegamos al punto final, terminar
            if x == x1 and y == y1:
                break
            
            # Calcular el error y ajustar las coordenadas
            e2 = 2 * err
            
            if e2 > -dy:
                err -= dy
                x += sx
            
            if e2 < dx:
                err += dx
                y += sy

    def glTriangle(self, A, B, C):
        self.glLine(A, B, self.currColor)
        self.glLine(B, C, self.currColor)
        self.glLine(C, A, self.currColor)

    def glRender(self):
        for model in self.models:
            self.activeModelMatrix = model.GetModelMatrix()
            self.activeVertexShader = model.vertexShader
            buffer = []

            for i in range(0, len(model.vertices), 3):
                x, y, z = model.vertices[i:i + 3]
                if self.activeVertexShader:
                    x, y, z = self.activeVertexShader([x, y, z], modelMatrix=self.activeModelMatrix)
                buffer.extend([x, y, z])

            self.glDrawPrimitives(buffer, 3, model)

    def glDrawPrimitives(self, buffer, vertexOffset, model):
        # Procesar cada triángulo (9 valores = 3 vértices * 3 coordenadas)
        triangleCount = len(buffer) // 9
        
        for i in range(triangleCount):
            # Obtener los 3 vértices del triángulo
            startIdx = i * 9
            A = buffer[startIdx:startIdx+3]
            B = buffer[startIdx+3:startIdx+6]
            C = buffer[startIdx+6:startIdx+9]

            # Obtener el color para este triángulo
            color = model.colors[i] if i < len(model.colors) else [1, 1, 1]
            shadedColor = model.fragmentShader(color=color)

            # Renderizar según el tipo de primitiva
            if self.primitiveType == POINTS:
                self.glPoint(A[0], A[1], shadedColor)
                self.glPoint(B[0], B[1], shadedColor)
                self.glPoint(C[0], C[1], shadedColor)
            elif self.primitiveType == LINES:
                # Para líneas, usar color blanco
                white_color = [1.0, 1.0, 1.0]
                self.glLine(A, B, white_color)
                self.glLine(B, C, white_color)
                self.glLine(C, A, white_color)
            elif self.primitiveType == TRIANGLES:
                self.glFillTriangle(A, B, C, shadedColor)

    def glFillTriangle(self, A, B, C, color):
        # Convertir a enteros
        x1, y1 = int(A[0]), int(A[1])
        x2, y2 = int(B[0]), int(B[1])
        x3, y3 = int(C[0]), int(C[1])
        
        # Ordenar vértices por Y (y1 <= y2 <= y3)
        if y1 > y2:
            x1, y1, x2, y2 = x2, y2, x1, y1
        if y2 > y3:
            x2, y2, x3, y3 = x3, y3, x2, y2
        if y1 > y2:
            x1, y1, x2, y2 = x2, y2, x1, y1
        
        # Si todos los puntos están en la misma línea horizontal, no renderizar
        if y1 == y3:
            return
        
        # Función para interpolar X basado en Y
        def get_x(y, x_start, y_start, x_end, y_end):
            if y_start == y_end:
                return x_start
            return x_start + (x_end - x_start) * (y - y_start) / (y_end - y_start)
        
        # Renderizar línea por línea
        for y in range(y1, y3 + 1):
            if y < 0 or y >= self.height:
                continue
            
            # Calcular las intersecciones X para este Y
            x_intersections = []
            
            # Lado A-C (lado largo)
            if y1 != y3:
                x_ac = get_x(y, x1, y1, x3, y3)
                x_intersections.append(x_ac)
            
            # Lado A-B o B-C dependiendo de la posición de Y
            if y <= y2:
                # Estamos en la parte superior, usar lado A-B
                if y1 != y2:
                    x_ab = get_x(y, x1, y1, x2, y2)
                    x_intersections.append(x_ab)
            else:
                # Estamos en la parte inferior, usar lado B-C
                if y2 != y3:
                    x_bc = get_x(y, x2, y2, x3, y3)
                    x_intersections.append(x_bc)
            
            # Si tenemos dos intersecciones, dibujar la línea horizontal
            if len(x_intersections) == 2:
                x_left = min(x_intersections)
                x_right = max(x_intersections)
                
                for x in range(int(x_left), int(x_right) + 1):
                    if 0 <= x < self.width:
                        self.glPoint(x, y, color)