import h5py
import numpy as np
import pyvista
import vtk

def read_vtkhdf(filename:str):
    reader = vtk.vtkHDFReader()
    reader.SetFileName(filename)
    reader.Update()    
    return pyvista.wrap(reader.GetOutput())

def c2f(array):
    return np.asfortranarray(np.transpose(array))

def f2c(array):
    return np.ascontiguousarray(np.transpose(array))

def read_slice(hdf5_file:h5py.File, var:str, zindex:int) -> np.ndarray:    
    dat = hdf5_file["VTKHDF"]["PointData"][var][zindex, :, :]
    return c2f(dat)

def init_vtkhdf(file:h5py.File, extent, origin=(0,0,0), spacing=(1,1,1),
                direction=(1, 0, 0, 0, 1, 0, 0, 0, 1)):
    vtkhdf_group = file.create_group("VTKHDF")
    vtkhdf_group.attrs.create("Version", [1, 0])
    vtkhdf_group.attrs.create("Type", np.string_("ImageData"))
    vtkhdf_group.attrs.create("WholeExtent", extent)
    vtkhdf_group.attrs.create("Origin", origin)
    vtkhdf_group.attrs.create("Spacing", spacing)
    vtkhdf_group.attrs.create("Direction", direction)
    vtkhdf_group.create_group("PointData")

def create_dataset(file:h5py.File, shape, name:str, order:str="F"):
    field_data_group = file["VTKHDF"]["PointData"]
    new_shape = shape[::-1] if order == "F" else shape
    chunk_shape = np.copy(new_shape)
    chunk_shape[0] = 1 # z will be at index 0
    field_data_group.attrs.create("Scalars", np.string_(name))   
    field_data_group.create_dataset(
        name, shape=new_shape, chunks=tuple(chunk_shape)
    )

def write_slice(file:h5py.File, array:np.ndarray, name:str,
                zindex:int):
    # paraview will be transposing it to read
    arr_c = f2c(array) if array.flags.f_contiguous else array
    file["VTKHDF"]["PointData"][name][zindex,:,:] = arr_c[np.newaxis,:,:]

def write_vtkhdf(filename:str, imagedata,
                 direction=(1, 0, 0, 0, 1, 0, 0, 0, 1),
                 **kwargs):
    if type(imagedata) is vtk.vtkImageData:
        imagedata:pyvista.ImageData = pyvista.wrap(imagedata)
    with h5py.File(filename, "w", **kwargs) as f:
        init_vtkhdf(f, imagedata.extent, origin=imagedata.origin,
                    spacing=imagedata.spacing, direction=direction)
        for var in imagedata.array_names:
            create_dataset(f, imagedata.dimensions, var)
            for i in range(imagedata.dimensions[2]):
                write_slice(f, get_imagedata(imagedata, var)[:,:,i], var, i)

def set_imagedata(image_data:pyvista.ImageData, array:np.ndarray, name:str):
    image_data[name] = array.flatten(order="F")

def get_imagedata(image_data:pyvista.ImageData, name:str):
    return image_data[name].reshape(image_data.dimensions, order="F")

def dimensions2extent(dimensions):
    return [x for v in dimensions for x in (0,v-1)]