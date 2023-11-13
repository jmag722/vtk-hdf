import h5py
import numpy as np

import vtkhdf.image as v5i

dimensions = (1200, 1501, 653) # 9.4 GB per 64-bit dataset!
spacing = (1e-3, 2e-3, 5e-4)
origin = v5i.origin_of_centered_image(dimensions, spacing, 2)

x,y,z = v5i.get_point_axes(dimensions, spacing, origin)

cache_slice_nbytes = dimensions[0] * dimensions[1] * 8
with h5py.File("mybigimage.hdf", "w", rdcc_nbytes=cache_slice_nbytes) as f:

    v5i.initialize(f, v5i.dimensions2extent(dimensions),
                   origin=origin, spacing=spacing)
    
    dset = v5i.create_point_dataset(f, "data")

    slice = np.empty(dimensions[:-1], order="F")
    for k, valz in enumerate(z):
        # avoid meshgrid with newaxis
        slice = np.sqrt(x[:, np.newaxis]**2 + y**2, order="F")
        v5i.write_slice(dset, slice, k)

with h5py.File("mybigimage.hdf", "r", rdcc_nbytes=cache_slice_nbytes) as f:
    dset = v5i.get_point_dataset(f, "data")
    slice = v5i.read_slice(dset, 42)
    assert np.allclose(slice, np.sqrt(x[:, np.newaxis]**2 + y**2, order="F"))
