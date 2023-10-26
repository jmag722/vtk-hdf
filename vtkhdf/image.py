import h5py
import numpy as np
from numpy.typing import ArrayLike
import pyvista
import vtk

"""
Python interface for VTK HDF ImageData format.

References & Examples Used:
https://vtk.org/doc/nightly/html/VTKHDFFileFormat.html
https://docs.vtk.org/en/latest/design_documents/VTKFileFormats.html#hdf-file-formats
VTK Issue #18956 and its discourse at 
    https://discourse.vtk.org/t/working-with-a-hdf-file-in-vtk/11233
    was helpful in setting up a working case.
https://www.kitware.com/developing-hdf5-readers-using-vtkpythonalgorithm/
"""

VTKHDF = "VTKHDF"
IMAGEDATA = "ImageData"
POINTDATA = "PointData"
VERSION = "Version"
TYPE = "Type"
EXTENT = "WholeExtent"
ORIGIN = "Origin"
SPACING = "Spacing"
DIRECTION = "Direction"
SCALARS = "Scalars"

def read_vtkhdf(filename:str):
    """
    Wrapper for reading VTK HDF file format. Currently, the vtk.vtkHDFReader
    does not support LZF compressed datasets (GZIP does work).

    Parameters
    ----------
    filename : str
        VTK HDF filename.

    Returns
    -------
    pyvista.DataSet
        The PyVista wrapped dataset.
    """    
    reader = vtk.vtkHDFReader()
    reader.SetFileName(filename)
    reader.Update()    
    return pyvista.wrap(reader.GetOutput())

def c2f_reshape(array:ArrayLike) -> np.ndarray:
    """
    Convert a row-major (C) array to column-major (Fortran).

    Parameters
    ----------
    array : ArrayLike
        C-order array

    Returns
    -------
    np.ndarray
        Transposed array in Fortran order
    """    
    return np.asfortranarray(np.transpose(array))

def f2c_reshape(array:ArrayLike) -> np.ndarray:
    """
    Convert a column-major (Fortran) array to row-major (C).

    Parameters
    ----------
    array : ArrayLike
        Fortran-order array

    Returns
    -------
    np.ndarray
        Transposed array in C order
    """
    return np.ascontiguousarray(np.transpose(array))

def read_slice(hdf5_file:h5py.File, var:str, index:int,
               f_contiguous:bool=True) -> np.ndarray:
    """
    Read ImageData slice from an HDF5 file. Slices are taken on the 
    last index of the corresponding column-major (F) ordered dataset.

    By default, slices are output in Fortran order to match 
    ImageData storage.

    Parameters
    ----------
    hdf5_file : h5py.File
        VTK HDF file
    var : str
        ImageData variable name
    index : int
        Index of the last axis (2, typically "z") to extract from ImageData.
    f_contiguous : bool
        Whether to return slice in Fortran or C, order,
        by default True (Fortran)  

    Returns
    -------
    np.ndarray
        ImageData slice
    """    
    dat = hdf5_file[VTKHDF][POINTDATA][var][index, :, :]
    return c2f_reshape(dat) if f_contiguous else dat

def initialize(file:h5py.File, extent:tuple, origin:tuple=(0,0,0),
               spacing:tuple=(1,1,1),
               direction:tuple=(1, 0, 0, 0, 1, 0, 0, 0, 1)):
    """
    Initialize a VTK HDF file group with correct attributes.

    Parameters
    ----------
    file : h5py.File
        VTK HDF file
    extent : tuple
        Extent of ImageData
    origin : tuple, optional
        Origin of ImageData, by default (0,0,0)
    spacing : tuple, optional
        Spacing of ImageData, by default (1,1,1)
    direction : tuple, optional
        Direction of ImageData axes, by default (1, 0, 0, 0, 1, 0, 0, 0, 1)
    """    
    vtkhdf_group = file.create_group(VTKHDF)
    vtkhdf_group.attrs.create(VERSION, [1, 0])
    vtkhdf_group.attrs.create(TYPE, np.string_(IMAGEDATA))
    vtkhdf_group.attrs.create(EXTENT, extent)
    vtkhdf_group.attrs.create(ORIGIN, origin)
    vtkhdf_group.attrs.create(SPACING, spacing)
    vtkhdf_group.attrs.create(DIRECTION, direction)
    vtkhdf_group.create_group(POINTDATA)

def create_dataset(file:h5py.File, name:str, **kwargs):
    """
    Create HDF5 dataset within file. Data is chunked by
    the slowest-changing/last index in the flattened Fortran array.

    Paraview (as of 5.11.2) and vtk.vtkHDFReader do not currently support LZF
    compression, but GZIP is supported.

    Parameters
    ----------
    file : h5py.File
        VTK HDF file
    name : str
        Dataset name to create
    """    
    field_data_group = file[VTKHDF][POINTDATA]
    # reverse what is given in WholeExtent so paraview can read
    shape_c = extent2dimensions(file[VTKHDF].attrs[EXTENT])[::-1]
    chunk_shape = (1, *shape_c[1:]) # single z-slices, z will be at index 0
    field_data_group.attrs.create(SCALARS, np.string_(name))   
    field_data_group.create_dataset(
        name, shape=shape_c, chunks=chunk_shape, **kwargs
    )

def write_slice(file:h5py.File, array:np.ndarray, name:str,
                index:int):
    """
    Write array slice to file.

    Parameters
    ----------
    file : h5py.File
        Opened VTK HDF file.
    array : np.ndarray
        Array to write to file
    name : str
        Dataset name to write to
    index : int
        ImageData slice index
    """    
    # paraview will be transposing it to read
    arr_c = f2c_reshape(array) if array.flags.f_contiguous else array
    file[VTKHDF][POINTDATA][name][index,:,:] = arr_c[np.newaxis,:,:]

def write_vtkhdf(h5_file:h5py.File, imagedata,
                 direction=(1, 0, 0, 0, 1, 0, 0, 0, 1),
                 **kwargs):
    """
    Wrapper for writing ImageData object to VTK HDF format. This 
    is mostly just for convenience and debugging purposes, as any
    ImageData requiring HDF formatting would be too big to have sitting
    in memory.

    Parameters
    ----------
    h5_file : h5py.File
        Opened HDF5 file to write to (assumed empty)
    imagedata : vtk.ImageData | pyvista.ImageData
        ImageData
    direction : tuple, optional
        ImageData direction, by default (1, 0, 0, 0, 1, 0, 0, 0, 1)
    """    
    if type(imagedata) is vtk.vtkImageData:
        imagedata:pyvista.ImageData = pyvista.wrap(imagedata)
    initialize(h5_file, imagedata.extent, origin=imagedata.origin,
               spacing=imagedata.spacing, direction=direction)
    for var in imagedata.array_names:
        create_dataset(h5_file, var, **kwargs)
        for i in range(imagedata.dimensions[2]):
            write_slice(h5_file, get_array(imagedata, var)[:,:,i], var, i)

def set_array(image_data:pyvista.ImageData, array:np.ndarray, name:str):
    """
    Convenience function to write unflattened dataset to an ImageData
    object. Handles converting to column-major order.

    Parameters
    ----------
    image_data : pyvista.ImageData
        ImageData
    array : np.ndarray
        3D array to assign as ImageData dataset
    name : str
        New dataset name
    """    
    image_data[name] = array.flatten(order="F")

def get_array(image_data:pyvista.ImageData, name:str):
    """
    Convenience function to get unflattened dataset from an ImageData
    object. Handles converting to column-major order.

    Parameters
    ----------
    image_data : pyvista.ImageData
        ImageData object
    name : str
        Dataset name to get.

    Returns
    -------
    np.ndarray
        Entire dataset from ImageData
    """    
    return image_data[name].reshape(image_data.dimensions, order="F")

def dimensions2extent(dimensions:tuple):
    """
    Get ImageData extent from given dimensions.

    Parameters
    ----------
    dimensions : tuple
        Dimensions of array or ImageData object

    Returns
    -------
    tuple
        Array extents
    """    
    return tuple(x for v in dimensions for x in (0,v-1))

def extent2dimensions(extent:tuple):
    """
    Compute dataset dimensions from ImageData extent.

    Parameters
    ----------
    extent : tuple
        Extent of array or ImageData

    Returns
    -------
    tuple
        Dimensions of array or ImageData
    """    
    return tuple(x+1 for x in extent[1::2])

def axis_length(dimensions:tuple, spacing:tuple):
    """
    Length of an array from its dimensions
    and constant spacing.

    Parameters
    ----------
    dimensions : tuple
        Array dimensions
    spacing : tuple
        Array spacing in each dimension

    Returns
    -------
    np.ndarray
        Array length in every dimension
    """    
    return (np.array(dimensions)-1)*np.array(spacing)

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
    origin = -0.5*axis_length(dimensions, spacing)
    if zero_last_axis and origin.size == 3:
        origin[2] = 0.0
    return origin

def get_axis(dimension:float, spacing:float, origin:float) -> np.ndarray:
    """
    Get position of points along axis.

    Parameters
    ----------
    dimension : float
        Number of points
    spacing : float
        Spacing between points
    origin : float
        Origin of array of points

    Returns
    -------
    np.ndarray
        Array of positional points for the given dimension.
    """    
    return np.linspace(origin,
                       origin + axis_length(dimension, spacing),
                       dimension,
                       endpoint=True)

def get_axes(dimensions:tuple, spacing:tuple, origin:tuple):
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
    return tuple([get_axis(dim, spacing[i], origin[i])
                  for i,dim in enumerate(dimensions)])

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