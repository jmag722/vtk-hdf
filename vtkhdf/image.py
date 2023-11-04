import h5py
import numpy as np
from numpy.typing import ArrayLike
import pyvista
import vtk

import vtkhdf.image_utils as iu

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
CELLDATA = "CellData"
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
    vtk.vtkImageData
        ImageData output
    """    
    reader = vtk.vtkHDFReader()
    reader.SetFileName(filename)
    reader.Update()
    return reader.GetOutput()

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

def get_dataset(h5_file:h5py.File, var:str) -> h5py.Dataset:
    """
    Get dataset from VTK HDF file, either point or cell data.
    Note the data has NOT been transposed to Fortran order.

    Parameters
    ----------
    h5_file : h5py.File
        opened VTK HDF file to read
    var : str
        Variable to return. Variable can represent either point or cell data

    Returns
    -------
    h5py.Dataset
        ImageData dataset to return, either cell or point data
    """    
    return (
        get_point_dataset(h5_file, var) if var in h5_file[VTKHDF][POINTDATA]
        else get_cell_dataset(h5_file, var)
    )

def get_point_dataset(h5_file:h5py.File, var:str) -> h5py.Dataset:
    """
    Get point dataset from VTK HDF file. Note the data has NOT
    been transposed to Fortran order.

    Parameters
    ----------
    h5_file : h5py.File
        opened VTK HDF file to read
    var : str
        Variable to return. Variable must be point data

    Returns
    -------
    h5py.Dataset
        ImageData point dataset
    """    
    return h5_file[VTKHDF][POINTDATA][var]

def get_cell_dataset(h5_file:h5py.File, var:str) -> h5py.Dataset:
    """
    Get cell dataset from VTK HDF file. Note the data has NOT
    been transposed to Fortran order.

    Parameters
    ----------
    h5_file : h5py.File
        opened VTK HDF file to read
    var : str
        Variable to return. Variable must be cell data

    Returns
    -------
    h5py.Dataset
        ImageData cell dataset
    """
    return h5_file[VTKHDF][CELLDATA][var]

def read_slice(dset:h5py.Dataset, index:int,
               f_contiguous:bool=True) -> np.ndarray:
    """
    Read 2D ImageData slice from VTK HDF dataset. Slices are taken on the 
    last index of the corresponding column-major (F) ordered dataset.

    By default, slices are output in Fortran order to match 
    ImageData storage.

    Parameters
    ----------
    dset : h5py.Dataset
        Dataset to read slice from (point or cell data)
    index : int
        Index of the last axis (2, typically "z") to extract from ImageData.
    f_contiguous : bool
        Whether to return slice in Fortran or C, order,
        by default True (Fortran)  

    Returns
    -------
    np.ndarray
        2D (last index squeezed) ImageData slice
    """
    dat = dset[index, :, :]
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
    vtkhdf_group.create_group(CELLDATA)

def create_point_dataset(h5_file:h5py.File, var:str, **kwargs) -> h5py.Dataset:
    """
    Create HDF5 point dataset within file. Data is chunked by
    the slowest-changing/last index in the Fortran array.

    Paraview (as of 5.11.2) and vtk.vtkHDFReader do not currently support LZF
    compression, but GZIP is supported.

    Parameters
    ----------
    h5_file : h5py.File
        opened VTK HDF file for read
    var : str
        Point dataset variable name

    Returns
    -------
    h5py.Dataset
        Point dataset
    """    
    group = h5_file[VTKHDF][POINTDATA]
    shape_c = get_point_data_shape(h5_file)
    chunk_shape = get_chunk_shape(shape_c)
    group.attrs.create(SCALARS, np.string_(var))   
    return group.create_dataset(
        var, shape=shape_c, chunks=chunk_shape, **kwargs
    )

def get_point_data_shape(h5_file:h5py.File) -> tuple:
    """
    Get the shape of the point dataset as it will be written to HDF.
    Note that this is the reverse of what is given by WholeExtent, because
    Paraview expects the array to be in C-order.

    Parameters
    ----------
    h5_file : h5py.File
        open VTK HDF file for read

    Returns
    -------
    tuple
        shape of point dataset as it will be written to file
    """    
    # reverse what is given in WholeExtent so paraview can read
    return extent2dimensions(h5_file[VTKHDF].attrs[EXTENT])[::-1]

def get_chunk_shape(shape_c:tuple) -> tuple:
    """
    Get the chunk shape for the point or cell array data as it
    is written to the HDF5 file (C-order). This means that the last or sliced
    index (axis 2 in Fortran-order) is now the first index here.

    Parameters
    ----------
    shape_c : tuple
        Shape of point or cell dataset, in transposed C-order

    Returns
    -------
    tuple
        chunk shape for given dataset shape
    """
    return (1, *shape_c[1:])

def create_cell_dataset(h5_file:h5py.File, var:str, **kwargs) -> h5py.Dataset:
    """
    Create HDF5 cell dataset within file. Data is chunked by
    the slowest-changing/last index in the Fortran array.

    Paraview (as of 5.11.2) and vtk.vtkHDFReader do not currently support LZF
    compression, but GZIP is supported.

    Parameters
    ----------
    h5_file : h5py.File
        opened VTK HDF file to read
    var : str
        Cell dataset variable to initialize

    Returns
    -------
    h5py.Dataset
        Cell dataset
    """    
    group = h5_file[VTKHDF][CELLDATA]
    shape_c = get_cell_data_shape(h5_file)
    chunk_shape = get_chunk_shape(shape_c)
    group.attrs.create(SCALARS, np.string_(var))   
    return group.create_dataset(
        var, shape=shape_c, chunks=chunk_shape, **kwargs
    )

def get_cell_data_shape(h5_file:h5py.File):
    """
    Get the shape of the cell dataset as it will be written to HDF.
    Note that this is the reverse of what is given by WholeExtent
    (and decremented by one for cell data), because Paraview expects
    the array to be in C-order.

    Parameters
    ----------
    h5_file : h5py.File
        open VTK HDF file for read

    Returns
    -------
    tuple
        shape of point dataset as it will be written to file
    """
    return iu.point2cell_dimension(get_point_data_shape(h5_file))

def write_slice(dset:h5py.Dataset, array:np.ndarray, index:int):
    """
    Write array slice to file.

    Parameters
    ----------
    file : h5py.Dataset
        VTK HDF dataset (point or cell data)
    array : np.ndarray
        2D array to write to file, in either C or F-order
    index : int
        Slice index corresponding to the last axis of the ImageData dataset
    """    
    # paraview will be transposing it to read
    arr_c = f2c_reshape(array) if array.flags.f_contiguous else array
    dset[index,:,:] = arr_c[np.newaxis,:,:]

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
    if isinstance(imagedata, vtk.vtkImageData):
        imagedata:pyvista.ImageData = pyvista.wrap(imagedata)
    initialize(h5_file, imagedata.extent, origin=imagedata.origin,
               spacing=imagedata.spacing, direction=direction)
    for var in imagedata.array_names:
        if var in imagedata.point_data.keys():
            dset = create_point_dataset(h5_file, var, **kwargs)
            nslices = imagedata.dimensions[2]
            arr = get_point_array(imagedata, var)
        elif var in imagedata.cell_data.keys():
            dset = create_cell_dataset(h5_file, var, **kwargs)
            arr = get_cell_array(imagedata, var)
            nslices = iu.point2cell_dimension(imagedata.dimensions)[2]
        for i in range(nslices):
            write_slice(dset, arr[:,:,i], i)

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
    Compute ImageData point dataset dimensions from ImageData extent.

    Parameters
    ----------
    extent : tuple
        Extent of array or ImageData

    Returns
    -------
    tuple
        Dimensions of array or ImageData point data
    """    
    return tuple(ub-lb+1 for (lb, ub) in list(zip(extent[0::2], extent[1::2])))

def extent2cellshape(extent:tuple):
    """
    Get the dimensional shape of the cell data from the 
    corresponding ImageData extent (which is the extent of point datasets)

    Parameters
    ----------
    extent : tuple
        Extent of ImageData

    Returns
    -------
    tuple
        Dimensions of ImageData cell data
    """    
    return tuple(ub-lb if ub-lb > 0 else 1
                 for (lb, ub) in list(zip(extent[0::2], extent[1::2])))

def set_point_array(image_data:pyvista.ImageData, array:np.ndarray, var:str):
    """
    Convenience function to write unflattened point dataset to an ImageData
    object. Handles conversion from column-major order.

    Parameters
    ----------
    image_data : pyvista.ImageData
        ImageData
    array : np.ndarray
        3D array to assign as ImageData point dataset
    var : str
        New point dataset variable name
    """    
    image_data.point_data[var] = array.flatten(order="F")

def set_cell_array(image_data:pyvista.ImageData, array:np.ndarray, var:str):
    """
    Convenience function to write unflattened point dataset to an ImageData
    object. Handles conversion from column-major order.

    Parameters
    ----------
    image_data : pyvista.ImageData
        ImageData
    array : np.ndarray
        3D array to assign as ImageData cell dataset
    var : str
        New cell dataset variable name
    """    
    image_data.cell_data[var] = array.flatten(order="F")

def get_point_array(image_data:pyvista.ImageData, var:str):
    """
    Convenience function to get unflattened point dataset
    from an ImageData object. Handles converting to column-major order.

    Parameters
    ----------
    image_data : pyvista.ImageData
        ImageData object
    var : str
        Point dataset variable key to get.

    Returns
    -------
    np.ndarray
        Entire point dataset from ImageData
    """    
    return image_data.point_data[var].reshape(image_data.dimensions, order="F")

def get_cell_array(image_data:pyvista.ImageData, var:str):
    """
    Convenience function to get unflattened cell dataset
    from an ImageData object. Handles converting to column-major order.

    Parameters
    ----------
    image_data : pyvista.ImageData
        ImageData object
    var : str
        Cell dataset variable key to get.

    Returns
    -------
    np.ndarray
        Entire cell dataset from ImageData
    """    
    return image_data.cell_data[var].reshape(extent2cellshape(image_data.extent),
                                             order="F")