import h5py
import numpy as np
import pyvista
import vtk

def read_vtkhdf(filename:str):
    reader = vtk.vtkHDFReader()
    reader.SetFileName(filename)
    reader.Update()    
    return pyvista.wrap(reader.GetOutput())

def read_slice(hdf5_file:h5py.File, var:str, zindex:int) -> np.ndarray:    
    dat = hdf5_file["VTKHDF"]["PointData"][var][zindex, :, :]
    return np.asfortranarray(np.transpose(dat))

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

def create_dataset(file:h5py.File, shape, name:str):
    field_data_group = file["VTKHDF"]["PointData"]
    new_shape = shape[::-1]
    chunk_shape = np.copy(new_shape)
    chunk_shape[0] = 1 # z will be at index 0
    field_data_group.attrs.create("Scalars", np.string_(name))   
    field_data_group.create_dataset(
        name, shape=new_shape, chunks=tuple(chunk_shape)
    )

def write_slice(file:h5py.File, array:np.ndarray, name:str,
                zindex:int):
    field_data_group = file["VTKHDF"]["PointData"]
    if not array.flags.f_contiguous:
        raise TypeError("Input must be column-major (Fortran) array.")
    # transpose b/c paraview will be transposing it to read
    arr_c = np.transpose(array)
    field_data_group[name][zindex,:,:] = arr_c[np.newaxis,:,:]

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
                write_slice(f, get_var(imagedata, var)[:,:,i], var, i)

def set_var(image_data:pyvista.ImageData, array:np.ndarray, name:str):
    image_data[name] = array.flatten(order="F")

def get_var(image_data:pyvista.ImageData, name:str):
    return image_data[name].reshape(image_data.dimensions, order="F")