import os

def load_obj_with_mtl(filename):
    vertices = []
    texcoords = []
    normals = []
    faces = []
    materials = {}
    material_faces = []
    current_material = None

    base_path = os.path.dirname(filename)

    with open(filename, 'r') as file:
        for line in file:
            if line.startswith('v '):
                parts = line.strip().split()
                vertices.append(tuple(map(float, parts[1:4])))
            elif line.startswith('vt '):
                parts = line.strip().split()
                texcoords.append(tuple(map(float, parts[1:3])))
            elif line.startswith('vn '):
                parts = line.strip().split()
                normals.append(tuple(map(float, parts[1:4])))
            elif line.startswith('f '):
                parts = line.strip().split()[1:]
                face = []
                for part in parts:
                    vals = part.split('/')
                    v = int(vals[0]) - 1
                    t = int(vals[1]) - 1 if len(vals) > 1 and vals[1] else None
                    n = int(vals[2]) - 1 if len(vals) > 2 and vals[2] else None
                    face.append((v, t, n))
                faces.append(face)
                material_faces.append(current_material)
            elif line.startswith('mtllib'):
                mtl_file = line.strip().split()[1]
                mtl_path = os.path.join(base_path, mtl_file)
                materials = load_mtl(mtl_path)
            elif line.startswith('usemtl'):
                current_material = line.strip().split()[1]

    # Retornar tupla extendida con materiales asociados por cara
    return vertices, list(zip(faces, material_faces)), materials, texcoords

def load_mtl(filename):
    contents = {}
    current_mat = None
    base_path = os.path.dirname(filename)

    with open(filename, 'r') as f:
        for line in f:
            if line.startswith('newmtl'):
                current_mat = line.strip().split()[1]
                contents[current_mat] = {'name': current_mat}
            elif line.startswith('map_Kd') and current_mat:
                texture_path = line.strip().split()[-1]
                texture_path = texture_path.replace("tEXTURE\\", "").replace("TEXTURE\\", "")
                contents[current_mat]['map_Kd'] = os.path.join(base_path, "tEXTURE", os.path.basename(texture_path))

    return contents
