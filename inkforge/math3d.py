import numpy as np


def normalize(v):
    v = np.asarray(v, dtype=np.float32)
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-12)


def look_at(eye, target, up=(0, 1, 0)):
    z = normalize(np.asarray(eye) - target)
    if abs(np.dot(z, up)) > .99:
        up = (0, 0, 1)
    x = normalize(np.cross(up, z))
    y = np.cross(z, x)
    m = np.eye(4, dtype=np.float32)
    m[:3, :3] = np.array([x, y, z])
    m[:3, 3] = -m[:3, :3] @ eye
    return m


def perspective(fov, aspect, near, far):
    f = 1 / np.tan(np.radians(fov) / 2)
    return np.array([[f/aspect, 0, 0, 0], [0, f, 0, 0],
                     [0, 0, far/(near-far), near*far/(near-far)],
                     [0, 0, -1, 0]], dtype=np.float32)


def orthographic(left, right, bottom, top, near, far):
    return np.array([[2/(right-left), 0, 0, -(right+left)/(right-left)],
                     [0, 2/(top-bottom), 0, -(top+bottom)/(top-bottom)],
                     [0, 0, 1/(near-far), near/(near-far)], [0, 0, 0, 1]], dtype=np.float32)


def node_matrix(node):
    if node.matrix:
        return np.asarray(node.matrix, dtype=np.float32).reshape(4, 4).T
    x, y, z, w = node.rotation or (0, 0, 0, 1)
    r = np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                  [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                  [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]], dtype=np.float32)
    m = np.eye(4, dtype=np.float32)
    m[:3, :3] = r @ np.diag(node.scale or (1, 1, 1))
    m[:3, 3] = node.translation or (0, 0, 0)
    return m
