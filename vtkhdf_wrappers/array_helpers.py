import numpy as np

def compute_origin(dimensions, spacing,
                   zero_z:bool=True) -> np.ndarray:
    origin = -0.5*(np.array(dimensions)-1)*np.array(spacing)
    if zero_z and origin.size == 3:
        origin[2] = 0.0
    return origin

def compute_axis_array(dim:float, step:float, origin:float) -> np.ndarray:
    return np.linspace(origin, origin + (dim-1)*step, dim,
                       endpoint=True)

def compute_axis_arrays(dimensions, spacing, origin):
    x = compute_axis_array(dimensions[0], spacing[0], origin[0])
    y = compute_axis_array(dimensions[1], spacing[1], origin[1])
    z = compute_axis_array(dimensions[2], spacing[2], origin[2])
    return x,y,z

def mesh(x,y,z):
    return np.meshgrid(x, y, z, indexing="ij")