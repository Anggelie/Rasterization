def load_obj(filename):
    vertices = []
    faces = []

    with open(filename, 'r') as file:
        for line in file:
            if line.startswith('v '):
                parts = line.strip().split()
                x, y, z = map(float, parts[1:4])
                vertices.append((x, y, z))
            elif line.startswith('f '):
                parts = line.strip().split()
                face = []
                for part in parts[1:]:
                    if '/' in part:
                        idx = part.split('/')[0]
                        face.append(int(idx) - 1)
                    else:
                        face.append(int(part) - 1)
                faces.append(face)
    return vertices, faces

def test():
    vertices, faces = load_obj('C:/Users/angge/Downloads/Rasterizer2025/Rasterizer2025/Rasterizer2025/Fig/girlOBJ.obj')
    print(vertices)
    print(faces)

if __name__ == '__main__':
    test()
