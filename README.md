# vtk-hdf
Python interface to read and write the VTK HDF format.

## Installation & Paraview
Dependencies are given in the [requirements](requirements.txt) file.

VTK HDF format is relatively new, and will require a recent vtk library to support it. For visualization, this means you'll need Paraview 5.10 or later for basic VTK HDF support. However, for using cell data with the ImageData HDF format (as an example), I needed a nightly build of Paraview in order to get the cell data visualization to work (the latest 5.11.2 release had not worked). Paraview 5.12 should contain the necessary changes to support the entire standard.

## Quick Start
### Example 1: Reading and writing entire ImageData objects at once
In the most trivial case, assume you have an ImageData object (vtk.ImageData or pyvista.ImageData) that you'd like to write to HDF5. Such an object could be initialized as:
```python
...
import vtkhdf.image as v5i

dimensions = (91, 51, 121)
spacing = (.01, .013, .03)
origin = v5i.origin_of_centered_image(dimensions, spacing, 2)
box = pyvista.ImageData(
    dimensions=dimensions,
    spacing=spacing,
    origin=origin
)
```
We now have an ImageData object, but it's empty. Let's assign a dataset to it:
```python
# dataset small enough that we can get away with meshgrid
X,Y,_ = v5i.mesh_axes(*v5i.get_point_axes(box.dimensions, box.spacing, box.origin))
data = X*X+Y*Y # positionally-dependent array data
v5i.set_point_array(box, data, "data")
```
This instance could easily be saved using `pyvista.DataObject.save`, but we can also
write the data in HDF5 format.
```python
with h5py.File("myimage.vtkhdf", "w") as f:
    v5i.write_vtkhdf(f, box)
```
We can verify that our data was saved correctly by reading it back for comparison
or viewing it in Paraview.
```python
mesh = v5i.read_vtkhdf("myimage.vtkhdf")
```
### Example 2: Writing large datasets by slice
Let's assume we'll be working with a much larger ImageData set.
```python
...
import vtkhdf.image as v5i

dimensions = (1200, 1501, 653) # 9.4 GB per 64-bit dataset!
spacing = (1e-3, 2e-3, 5e-4)
origin = v5i.origin_of_centered_image(dimensions, spacing, 2)
x,y,z = v5i.get_point_axes(dimensions, spacing, origin)
```
While many modern machines could hold this contiguous dataset in memory, often we don't need to and it will make our program more memory-efficient if we don't. Instead, we will initialize and save this ImageData slice-by-slice.

We'll open an HDF file for writing and set a cache size equal to a single slice that we'll be working with. Then we'll initialize the file to hold the 3D ImageData, though we haven't created it yet:
```python
cache_slice_nbytes = dimensions[0] * dimensions[1] * 8
with h5py.File("mybigimage.vtkhdf", "w", rdcc_nbytes=cache_slice_nbytes) as f:
    v5i.initialize(f, v5i.dimensions2extent(dimensions),
                   origin=origin, spacing=spacing)
    dset = v5i.create_point_dataset(f, "data")
```
As VTK uses column-major ordering (often called Fortran ordering), the data will be sliced by axis 2, the last index of the dataset. Assuming default ImageData direction, this is the "z" axis (though this could easily be changed for "x" or "y").
```python
    slice = np.empty(dimensions[:-1], order="F")
    for k, valz in enumerate(z):
        # avoid meshgrid with newaxis
        slice = np.sqrt(x[:, np.newaxis]**2 + y**2, order="F")
        v5i.write_slice(dset, slice, k)
```
**Note**: If the user was working with C-order numpy arrays, the dimensions, origin, and spacing input to `v5i.initialize` must be reversed from that of the C-order arrays. No modification needs to be made to the C-arrays themselves: `v5i.write_slice` handles transposing of the data when needed. Slices read via `v5i.read_slice` can be output in F or C-order according to the user's needs.

Now that we've written to this large file, we can access it later by slice as needed (or all at once if possible).
```python
with h5py.File("mybigimage.vtkhdf", "r", rdcc_nbytes=cache_slice_nbytes) as f:
    dset = v5i.get_point_dataset(f, "data")
    slice = v5i.read_slice(dset, 42)
    assert np.allclose(slice, np.sqrt(x[:, np.newaxis]**2 + y**2, order="F"))
```

Full examples can be found [here](./examples/).

## Further Reading

- [VTK File Formats](https://docs.vtk.org/en/latest/design_documents/VTKFileFormats.html#vtkhdf-file-format)
- [vtkHDFReader Class Reference](https://vtk.org/doc/nightly/html/classvtkHDFReader.html)
- [vtkHDFWriter Class Reference](https://vtk.org/doc/nightly/html/classvtkHDFWriter.html)
