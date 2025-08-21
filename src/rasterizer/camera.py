from .math3d import look_at, perspective

class Camera:
    def __init__(self, eye, target, up=(0,1,0), fov=60.0, near=0.1, far=100.0, aspect=16/9):
        self.eye = eye; self.target = target; self.up = up
        self.fov=fov; self.near=near; self.far=far; self.aspect=aspect

    @property
    def V(self): return look_at(self.eye, self.target, self.up)

    @property
    def P(self): return perspective(self.fov, self.aspect, self.near, self.far)