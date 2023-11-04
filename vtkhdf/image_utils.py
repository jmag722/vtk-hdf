import numpy as np
from numpy.typing import ArrayLike

def point2cell_dimension(dimensions:tuple) -> tuple:
    """
    Get the number of cell centers for point dimensions

    Parameters
    ----------
    dimensions : tuple
        point dimensions

    Returns
    -------
    tuple
        number of cell centers in each axis
    """    
    return tuple([d-1 if d>1 else d for d in dimensions])

def point2cell_origin(dimension:int, spacing:float, origin:float) -> float:
    """
    Get the position of the first cell center along an ImageData axis

    Parameters
    ----------
    dimension : int
        ImageData dimension
    spacing : float
        ImageData spacing
    origin : float
        ImageData origin

    Returns
    -------
    float
        cell center position
    """    
    return origin+0.5*spacing if dimension>1 else origin

def axis_length(dimensions:int, spacing:float) -> float:
    """
    Length of a 1D array from its dimension and constant spacing.

    Parameters
    ----------
    dimensions : int
        Array dimension
    spacing : float
        Array spacing

    Returns
    -------
    float
        Array length in every dimension
    """
    return (dimensions-1)*spacing

def origin_of_centered_image(dimensions:tuple, spacing:tuple,
                             zero_last_axis:bool=True) -> np.ndarray:
    """
    Obtain the origin for ImageData object (bottom left corner) assuming
    (0,0,0) is located at the ImageData centroid. Axis 2 (typically "z")
    is centered at the bottom slice by default, but can be centered at the
    middle as in x and y.

    Parameters
    ----------
    dimensions : tuple
        Number of dimensions
    spacing : tuple
        Spacing in each dimension
    zero_last_axis : bool, optional
        Whether axis 2 ("z") should be centered at the bottom plane or middle,
            by default True (bottom plane)

    Returns
    -------
    np.ndarray
        Origin of ImageData for (0,0,0) to be centered
    """    
    origin = -0.5*axis_length(np.array(dimensions), np.array(spacing))
    if zero_last_axis and origin.size == 3:
        origin[2] = 0.0
    return origin

def _get_axis(npts:int, spacing:float, offset:float) -> np.ndarray:
    """
    Get position of points along axis.

    Parameters
    ----------
    npts : int
        Number of points
    spacing : float
        Spacing between points
    offset : float
        Starting value of array of points

    Returns
    -------
    np.ndarray
        Array of positional points for the given dimension.
    """    
    return np.linspace(offset,
                       offset + axis_length(npts, spacing),
                       npts,
                       endpoint=True)

def get_point_axis(dimension:int, spacing:float, origin:float) -> np.ndarray:
    """
    Get 1D constant spacing of points for the dimensions, spacing,
    and origin of an ImageData axis.

    Parameters
    ----------
    dimension : int
        ImageData dimension
    spacing : float
        ImageData spacing
    origin : float
        ImageData origin

    Returns
    -------
    np.ndarray
        ImageData point positions
    """
    return _get_axis(dimension, spacing, origin)

def get_cell_axis(dimension:int, spacing:float, origin:float) -> np.ndarray:
    """
    Get 1D constant spacing of cell centers for the dimensions, spacing,
    and origin of an ImageData axis.

    Parameters
    ----------
    dimension : int
        ImageData dimension
    spacing : float
        ImageData spacing
    origin : float
        ImageData origin

    Returns
    -------
    np.ndarray
        cell center position array
    """    
    return _get_axis(point2cell_dimension([dimension])[0], spacing,
                     point2cell_origin(dimension, spacing, origin))

def get_point_axes(dimensions:tuple, spacing:tuple, origin:tuple):
    """
    Get one-dimensional axes for each dimension. Can be handy
    when creating datasets in ImageData based upon position.

    Parameters
    ----------
    dimensions : tuple
        Number of points in each dimension
    spacing : tuple
        Spacing in each dimension
    origin : tuple
        Origin in each dimension

    Returns
    -------
    tuple
        One-dimensional axes arrays for each dimension
    """    
    return tuple([get_point_axis(d, s, o)
                  for d,s,o in zip(dimensions, spacing, origin)])

def get_cell_axes(dimensions:tuple, spacing:tuple, origin:tuple):
    """
    Get one-dimensional cell axes for each dimension. Can be handy
    when creating cell datasets in ImageData based upon position.

    Parameters
    ----------
    dimensions : tuple
        Number of points in each dimension
    spacing : tuple
        Spacing in each dimension
    origin : tuple
        Origin in each dimension

    Returns
    -------
    tuple
        One-dimensional axes arrays for each dimension
    """    
    return tuple([get_cell_axis(d, s, o)
                  for d,s,o in zip(dimensions, spacing, origin)])

def mesh_axes(*xi:ArrayLike, indexing:str="ij", **kwargs):
    """
    Convenient wrapper for np.meshgrid() with indexing
    set to "ij".

    Parameters
    ----------
    xi : ArrayLike
        1D arrays

    Returns
    -------
    np.ndarray
        *xi as meshgrid
    """    
    return np.meshgrid(*xi, indexing=indexing, **kwargs)