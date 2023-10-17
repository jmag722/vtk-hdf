import h5py
import numpy as np
import pyvista
import vtk

def read_vtkhdf(filename:str):
    reader = vtk.vtkHDFReader()
    reader.SetFileName(filename)
    reader.Update()    
    return pyvista.wrap(reader.GetOutput())

def read_slice(hdf5_file:h5py.File, var:str, zindex:int):    
    dat = hdf5_file["VTKHDF"]["PointData"][var][zindex, :, :]
    return np.asfortranarray(np.transpose(dat))

def write_vtkhdf(filename:str, imagedata,
                 direction=(1, 0, 0, 0, 1, 0, 0, 0, 1),
                 **kwargs):
    if type(imagedata) is vtk.vtkImageData:
        imagedata:pyvista.ImageData = pyvista.wrap(imagedata)
    with h5py.File(filename, "w", **kwargs) as f:
        vtkhdf_group = f.create_group("VTKHDF")
        vtkhdf_group.attrs.create("Version", [1, 0])
        vtkhdf_group.attrs.create("Type", np.string_("ImageData"))
        vtkhdf_group.attrs.create("WholeExtent", imagedata.extent)
        vtkhdf_group.attrs.create("Origin", imagedata.origin)
        vtkhdf_group.attrs.create("Spacing", imagedata.spacing)
        vtkhdf_group.attrs.create("Direction", direction)

        field_data_group = vtkhdf_group.create_group("PointData")
        for var in imagedata.array_names:
            # transpose b/c paraview will be transposing it to read
            array = np.transpose(get_var(imagedata, var))
            new_shape = imagedata.dimensions[::-1]
            chunk_shape = np.copy(new_shape)
            chunk_shape[0] = 1 # z will be at index 0
            field_data_group.attrs.create("Scalars", np.string_(var))   
            dset = field_data_group.create_dataset(
                var, shape=new_shape, chunks=tuple(chunk_shape)
            )
            for i in range(imagedata.dimensions[2]):
                dset[i,:,:] = array[i,:,:]

def set_var(image_data:pyvista.ImageData, array:np.ndarray, name:str):
    image_data[name] = array.flatten(order="F")

def get_var(image_data:pyvista.ImageData, name:str):
    return image_data[name].reshape(image_data.dimensions, order="F")