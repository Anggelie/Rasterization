import numpy as np
from math import sin, cos, tan, radians

def normalize(v):
    v = np.asarray(v, dtype=np.float32)
    n = np.linalg.norm(v)
    return v / n if n > 1e-8 else v

def dot(a, b): return float(np.dot(a, b))
def cross(a, b): return np.cross(a, b)

def mat4_identity(): return np.eye(4, dtype=np.float32)

def translation(tx, ty, tz):
    M = mat4_identity(); M[0,3]=tx; M[1,3]=ty; M[2,3]=tz; return M

def scale(sx, sy=None, sz=None):
    if sy is None: sy = sx
    if sz is None: sz = sx
    M = mat4_identity(); M[0,0]=sx; M[1,1]=sy; M[2,2]=sz; return M

def rotation_x(deg):
    a=radians(deg); c,s=cos(a),sin(a); M=mat4_identity()
    M[1,1]=c; M[1,2]=-s; M[2,1]=s; M[2,2]=c; return M

def rotation_y(deg):
    a=radians(deg); c,s=cos(a),sin(a); M=mat4_identity()
    M[0,0]=c; M[0,2]=s; M[2,0]=-s; M[2,2]=c; return M

def rotation_z(deg):
    a=radians(deg); c,s=cos(a),sin(a); M=mat4_identity()
    M[0,0]=c; M[0,1]=-s; M[1,0]=s; M[1,1]=c; return M

def perspective(fov_deg, aspect, near, far):
    f = 1.0 / tan(radians(fov_deg)/2.0)
    M = np.zeros((4,4), dtype=np.float32)
    M[0,0]=f/aspect; M[1,1]=f
    M[2,2]=(far+near)/(near-far)
    M[2,3]=(2*far*near)/(near-far)
    M[3,2]=-1.0
    return M

def look_at(eye, target, up):
    eye=np.asarray(eye,np.float32); target=np.asarray(target,np.float32); up=np.asarray(up,np.float32)
    z = normalize(eye - target)     
    x = normalize(cross(up, z))      
    y = cross(z, x)                 
    M = mat4_identity()
    M[0,:3]=x; M[1,:3]=y; M[2,:3]=z
    T = mat4_identity(); T[:3,3] = -eye
    return M @ T
