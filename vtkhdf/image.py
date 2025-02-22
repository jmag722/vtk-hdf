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

Kolluru, Chaitanya. VTK Issue #18956 and its subsequent discussion.
    https://discourse.vtk.org/t/working-with-a-hdf-file-in-vtk/11233

Geveci, Berk. "Developing HDF5 readers using vtkPythonAlgorithm".
    https://www.kitware.com/developing-hdf5-readers-using-vtkpythonalgorithm/
"""

VTKHDF = "VTKHDF"
IMAGEDATA = "ImageData"
POINTDATA = "PointData"
CELLDATA = "CellData"
FIELDDATA = "FieldData"
VERSION = "Version"
TYPE = "Type"
EXTENT = "WholeExtent"
ORIGIN = "Origin"
SPACING = "Spacing"
DIRECTION = "Direction"
SCALARS = "Scalars"
EXTENSION = ".vtkhdf"


def initialize(file: h5py.File, extent: tuple, origin: tuple = (0, 0, 0),
               spacing: tuple = (1, 1, 1),
               direction: tuple = (1, 0, 0, 0, 1, 0, 0, 0, 1)):
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
    group = file.create_group(VTKHDF)
    group.attrs.create(VERSION, [1, 0])
    group.attrs.create(TYPE, np.bytes_(IMAGEDATA))
    group.attrs.create(EXTENT, extent)
    group.attrs.create(ORIGIN, origin)
    group.attrs.create(SPACING, spacing)
    group.attrs.create(DIRECTION, direction)
    group.create_group(POINTDATA)
    group.create_group(CELLDATA)
    group.create_group(FIELDDATA)


def read_slice(dset: h5py.Dataset, index: int,
               f_contiguous: bool = True) -> np.ndarray:
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


def write_slice(dset: h5py.Dataset, array: np.ndarray, index: int):
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
    dset[index, :, :] = arr_c[np.newaxis, :, :]


def create_point_dataset(h5_file: h5py.File, var: str, **kwargs) -> h5py.Dataset:
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
    return _create_dataset(group, shape=shape_c, var=var, **kwargs)


def create_cell_dataset(h5_file: h5py.File, var: str, **kwargs) -> h5py.Dataset:
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
    return _create_dataset(group, shape=shape_c, var=var, **kwargs)


def create_field_dataset(h5_file: h5py.File, var: str, **kwargs) -> h5py.Dataset:
    """
    Create ImageData FieldData in HDF format. Data arrays (not scalars)
    must be passed in with the `data` keyword argument.

    If the field data is a string type, it MUST be a byte string, not
    numpy bytes string. Using np.bytes_() doesn't fly with Paraview,
    use encode() instead (nuance with `numpy.bytes_` vs `bytes` type).

    Parameters
    ----------
    h5_file : h5py.File
        VTK HDF file, opened for write
    var : str
        variable name for field dataset

    Returns
    -------
    h5py.Dataset
        Field dataset
    """
    group = h5_file[VTKHDF][FIELDDATA]
    return group.create_dataset(var, **kwargs)


def _create_dataset(group: h5py.Group, shape: tuple, var: str, dtype=np.float64,
                    **kwargs) -> h5py.Dataset:
    """
    Create point or cell dataset with chunks based on the shape.

    Parameters
    ----------
    group : h5py.Group
        Point or cell data group
    shape : tuple
        shape of point or cell data
    var : str
        variable name
    dtype : str | np.dtype
        Data type for the dataset. Defaults to np.float64.

    Returns
    -------
    h5py.Dataset
        dataset created
    """
    chunk_shape = get_chunk_shape(shape)
    group.attrs.create(SCALARS, np.bytes_(var))
    return group.create_dataset(
        var, shape=shape, dtype=dtype, chunks=chunk_shape, **kwargs
    )


def get_point_data_shape(h5_file: h5py.File) -> tuple:
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


def get_cell_data_shape(h5_file: h5py.File):
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
    return point2cell_dimensions(get_point_data_shape(h5_file))


def get_chunk_shape(shape_c: tuple) -> tuple:
    """
    Get the chunk shape for the point or cell array data as it
    is written to the HDF5 file (C-order). This means that the last or sliced
    index (axis 2 in Fortran-order) is now the first index in the HDF file.

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


def get_dataset(h5_file: h5py.File, var: str) -> h5py.Dataset:
    """
    Get dataset from VTK HDF file, either point, cell, or field data.
    Note the data has NOT been transposed to Fortran order.

    Parameters
    ----------
    h5_file : h5py.File
        opened VTK HDF file to read
    var : str
        Variable to return. Variable can represent either point, cell
        or field data.

    Returns
    -------
    h5py.Dataset
        ImageData dataset to return, either point/cell/field data

    Raises
    ------
    KeyError
        If variable name is neither point, cell, or field data key
    """
    if var in h5_file[VTKHDF][POINTDATA]:
        return get_point_dataset(h5_file, var)
    elif var in h5_file[VTKHDF][CELLDATA]:
        return get_cell_dataset(h5_file, var)
    elif var in h5_file[VTKHDF][FIELDDATA]:
        return get_field_dataset(h5_file, var)
    else:
        raise KeyError(f"Key {var} is not found.")


def get_point_dataset(h5_file: h5py.File, var: str) -> h5py.Dataset:
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


def get_cell_dataset(h5_file: h5py.File, var: str) -> h5py.Dataset:
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


def get_field_dataset(h5_file: h5py.File, var: str):
    """
    Get field dataset from VTK HDF file.

    Parameters
    ----------
    h5_file : h5py.File
        opened VTK HDF file to read
    var : str
        Variable to return. Variable must be field data

    Returns
    -------
    h5py.Dataset
        ImageData field dataset
    """
    return h5_file[VTKHDF][FIELDDATA][var]


def read_vtkhdf(filename: str):
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


def write_vtkhdf(h5_file: h5py.File, imagedata,
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
        imagedata: pyvista.ImageData = pyvista.wrap(imagedata)

    initialize(h5_file, imagedata.extent, origin=imagedata.origin,
               spacing=imagedata.spacing, direction=direction)

    def _write_slice(dset, arr, i): return write_slice(dset, arr[:, :, i], i)

    for var in imagedata.point_data.keys():
        dset = create_point_dataset(h5_file, var, **kwargs)
        nslices = imagedata.dimensions[2]
        arr = get_point_array(imagedata, var)
        for i in range(nslices):
            _write_slice(dset, arr, i)

    for var in imagedata.cell_data.keys():
        dset = create_cell_dataset(h5_file, var, **kwargs)
        arr = get_cell_array(imagedata, var)
        nslices = point2cell_dimensions(imagedata.dimensions)[2]
        for i in range(nslices):
            _write_slice(dset, arr, i)

    for var in imagedata.field_data.keys():
        fdat = get_field_array(imagedata, var)
        create_field_dataset(h5_file, var, data=fdat, **kwargs)


def c2f_reshape(array: ArrayLike) -> np.ndarray:
    """
    Reshapes a row-major (C) array to column-major (Fortran).

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


def f2c_reshape(array: ArrayLike) -> np.ndarray:
    """
    Reshapes a column-major (Fortran) array to row-major (C).

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


def point2cell_dimensions(dimensions: tuple) -> tuple:
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
    return tuple([d-1 if d > 1 else d for d in dimensions])


def point2cell_extent(extent: tuple) -> tuple:
    """
    Get the extent of the cell centered data from point extent

    Parameters
    ----------
    extent : tuple
        point extent

    Returns
    -------
    tuple
        extent of cell center indices in each axis
    """
    return tuple([x for (lb, ub) in zip(extent[0::2], extent[1::2])
                  for x in (lb, ub-1 if ub > 0 else ub)])


def point2cell_origin(dimension: tuple, spacing: tuple, origin: tuple) -> tuple:
    """
    Get the position of the first cell center along an ImageData axis

    Parameters
    ----------
    dimension : tuple[int,...]
        ImageData dimension
    spacing : tuple[float,...]
        ImageData spacing
    origin : tuple[float,...]
        ImageData origin

    Returns
    -------
    tuple[float,...]
        cell center position
    """
    return tuple([o+0.5*s if d > 1 else o
                  for (d, s, o) in zip(dimension, spacing, origin)])


def dimensions2extent(dimensions: tuple):
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
    return tuple(x for v in dimensions for x in (0, v-1))


def extent2dimensions(extent: tuple):
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
    return tuple(ub-lb+1 for (lb, ub) in zip(extent[0::2], extent[1::2]))


def set_point_array(image_data: pyvista.ImageData, array: np.ndarray, var: str):
    """
    Convenience function to write unflattened point dataset to an ImageData
    object. Handles conversion from column-major order.

    Parameters
    ----------
    image_data : pyvista.ImageData
        ImageData
    array : np.ndarray
        3D array to assign as ImageData point dataset. Will be
        flattened in F-order.
    var : str
        New point dataset variable name
    """
    image_data.point_data[var] = array.flatten(order="F")


def set_cell_array(image_data: pyvista.ImageData, array: np.ndarray, var: str):
    """
    Convenience function to write unflattened point dataset to an ImageData
    object. Handles conversion from column-major order.

    Parameters
    ----------
    image_data : pyvista.ImageData
        ImageData
    array : np.ndarray
        3D array to assign as ImageData cell dataset. Will be
        flattened in F-order.
    var : str
        New cell dataset variable name
    """
    image_data.cell_data[var] = array.flatten(order="F")


def get_point_array(image_data: pyvista.ImageData, var: str):
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


def get_cell_array(image_data: pyvista.ImageData, var: str):
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
    cell_dims = extent2dimensions(point2cell_extent(image_data.extent))
    return image_data.cell_data[var].reshape(cell_dims, order="F")


def get_field_array(image_data: pyvista.ImageData, var: str):
    """
    Convenience function to get field dataset
    from an ImageData object.

    Handles converting fixed string array of pyvista (numpy bytes) to bytes
    object for variable length strings that Paraview seems to require.

    Parameters
    ----------
    image_data : pyvista.ImageData
        ImageData object
    var : str
        variable name

    Returns
    -------
    Any
        Array or list of FieldData
    """
    arr = image_data.field_data[var]
    if not isinstance(arr[0], str):
        return arr
    else:
        # using numpy.string_ here fails in Paraview
        return [v.encode("utf-8") for v in arr]


def origin_of_centered_image(dimensions: tuple, spacing: tuple,
                             zero_axis: int = None) -> tuple:
    """
    Obtain the origin for ImageData point data assuming
    (0,0,0) is located at the ImageData centroid. Any axis can be centered
    at the bottom plane (`zero_axis=...`) rather than the middle of the axis.

    Parameters
    ----------
    dimensions : tuple
        Number of points in each dimension
    spacing : tuple
        Spacing in each dimension
    zero_axis : int, optional
        Center this axis at the bottom image plane rather than at the
        middle of the axis, by default None (all axes centered at middle plane).

    Returns
    -------
    tuple
        Origin of ImageData for (0,0,0) to be at the ImageData centroid
    """
    origin = -0.5*axis_length(np.array(dimensions), np.array(spacing))
    if zero_axis is not None:
        if zero_axis >= origin.size:
            raise KeyError(f"Origin array of size {origin.size}"
                           f" cannot be indexed at {zero_axis}.")
        origin[zero_axis] = 0.0
    return tuple(origin)


def axis_length(dimensions: int, spacing: float) -> float:
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


def get_point_axis(dimension: int, spacing: float, origin: float) -> np.ndarray:
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


def get_cell_axis(dimension: int, spacing: float, origin: float) -> np.ndarray:
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
    return _get_axis(point2cell_dimensions([dimension])[0], spacing,
                     point2cell_origin([dimension], [spacing], [origin])[0])


def _get_axis(npts: int, spacing: float, offset: float) -> np.ndarray:
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


def get_point_axes(dimensions: tuple, spacing: tuple, origin: tuple):
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
                  for d, s, o in zip(dimensions, spacing, origin)])


def get_cell_axes(dimensions: tuple, spacing: tuple, origin: tuple):
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
                  for d, s, o in zip(dimensions, spacing, origin)])


def mesh_axes(*xi: ArrayLike, indexing: str = "ij", **kwargs):
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
