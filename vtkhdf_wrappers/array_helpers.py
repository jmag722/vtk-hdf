import numpy as np

def compute_origin(dimensions:np.ndarray, spacing:np.ndarray,
                   zero_z:bool=True):
    origin = -0.5*(dimensions-1)*spacing
    if zero_z:
        origin[2] = 0.0
    return origin

def compute_axis_array(d:float, s:float, o:float):
    return np.linspace(o, o + (d-1)*s, d, endpoint=True)

def compute_axis_arrays(dimensions, spacing, origin):
    x = compute_axis_array(dimensions[0], spacing[0], origin[0])
    y = compute_axis_array(dimensions[1], spacing[1], origin[1])
    z = compute_axis_array(dimensions[2], spacing[2], origin[2])
    return x,y,z

def mesh(x,y,z):
    return np.meshgrid(x, y, z, indexing="ij")