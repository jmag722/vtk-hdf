import h5py
import numpy as np
import pyvista
import vtk

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
    reader = vtk.vtkHDFReader()
    reader.SetFileName(filename)
    reader.Update()    
    return pyvista.wrap(reader.GetOutput())

def c2f_reshape(array):
    return np.asfortranarray(np.transpose(array))

def f2c_reshape(array):
    return np.ascontiguousarray(np.transpose(array))

def read_slice(hdf5_file:h5py.File, var:str, zindex:int) -> np.ndarray:    
    dat = hdf5_file[VTKHDF][POINTDATA][var][zindex, :, :]
    return c2f_reshape(dat)

def initialize(file:h5py.File, extent, origin=(0,0,0), spacing=(1,1,1),
                direction=(1, 0, 0, 0, 1, 0, 0, 0, 1)):
    vtkhdf_group = file.create_group(VTKHDF)
    vtkhdf_group.attrs.create(VERSION, [1, 0])
    vtkhdf_group.attrs.create(TYPE, np.string_(IMAGEDATA))
    vtkhdf_group.attrs.create(EXTENT, extent)
    vtkhdf_group.attrs.create(ORIGIN, origin)
    vtkhdf_group.attrs.create(SPACING, spacing)
    vtkhdf_group.attrs.create(DIRECTION, direction)
    vtkhdf_group.create_group(POINTDATA)

def create_dataset(file:h5py.File, name:str):
    field_data_group = file[VTKHDF][POINTDATA]
    # reverse what is given in WholeExtent so paraview can read
    shape_c = extent2dimensions(file[VTKHDF].attrs[EXTENT])[::-1]
    chunk_shape = (1, *shape_c[1:]) # single z-slices, z will be at index 0
    field_data_group.attrs.create(SCALARS, np.string_(name))   
    field_data_group.create_dataset(
        name, shape=shape_c, chunks=chunk_shape
    )

def write_slice(file:h5py.File, array:np.ndarray, name:str,
                zindex:int):
    # paraview will be transposing it to read
    arr_c = f2c_reshape(array) if array.flags.f_contiguous else array
    file[VTKHDF][POINTDATA][name][zindex,:,:] = arr_c[np.newaxis,:,:]

def write_vtkhdf(filename:str, imagedata,
                 direction=(1, 0, 0, 0, 1, 0, 0, 0, 1),
                 **kwargs):
    if type(imagedata) is vtk.vtkImageData:
        imagedata:pyvista.ImageData = pyvista.wrap(imagedata)
    with h5py.File(filename, "w", **kwargs) as f:
        initialize(f, imagedata.extent, origin=imagedata.origin,
                    spacing=imagedata.spacing, direction=direction)
        for var in imagedata.array_names:
            create_dataset(f, var)
            for i in range(imagedata.dimensions[2]):
                write_slice(f, get_array(imagedata, var)[:,:,i], var, i)

def set_array(image_data:pyvista.ImageData, array:np.ndarray, name:str):
    image_data[name] = array.flatten(order="F")

def get_array(image_data:pyvista.ImageData, name:str):
    return image_data[name].reshape(image_data.dimensions, order="F")

def dimensions2extent(dimensions):
    return tuple(x for v in dimensions for x in (0,v-1))

def extent2dimensions(extent):
    return tuple(x+1 for x in extent[1::2])

def axis_length(dimensions, spacing):
    return (np.array(dimensions)-1)*np.array(spacing)

def center_origin(dimensions, spacing,
                   zero_z:bool=True) -> np.ndarray:
    origin = -0.5*axis_length(dimensions, spacing)
    if zero_z and origin.size == 3:
        origin[2] = 0.0
    return origin

def get_axis(dimension:float, spacing:float, origin:float) -> np.ndarray:
    return np.linspace(origin,
                       origin + axis_length(dimension, spacing),
                       dimension,
                       endpoint=True)

def get_axes(dimensions, spacing, origin):
    return tuple([get_axis(dim, spacing[i], origin[i])
                  for i,dim in enumerate(dimensions)])

def mesh_axes(x,y,z):
    return np.meshgrid(x, y, z, indexing="ij")