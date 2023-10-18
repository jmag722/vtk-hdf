import pytest
import numpy as np
import pyvista
import h5py
import vtkhdf_wrappers.array_helpers as ah
import vtkhdf_wrappers.image_data as ida

@pytest.fixture
def radial_box():
    def _method():
        dimensions = np.array([151, 91, 113])
        spacing = np.array([.01, .011, .03])
        origin = ah.compute_origin(dimensions, spacing)
        box = pyvista.ImageData(
            dimensions=dimensions,
            spacing=spacing,
            origin=origin
        )
        X,Y,_ = ah.mesh(*ah.compute_axis_arrays(box.dimensions, box.spacing,
                                                box.origin))
        data = np.sqrt(X*X+Y*Y)
        ida.set_var(box, data, "data")
        return box
    return _method

def test_read_slice(tmp_path, radial_box):
    box = radial_box()
    ida.write_vtkhdf(tmp_path/"mybox-vti.hdf", box)
    h5_file = h5py.File(tmp_path/"mybox-vti.hdf", "r")
    for i in range(box.dimensions[2]):
        slice = ida.read_slice(h5_file, "data", i)
        assert slice.flags.f_contiguous
        assert slice.shape == box.dimensions[:-1]
        np.testing.assert_allclose(slice, ida.get_var(box, "data")[:,:,i])
    h5_file.close()

def test_init_vtkhdf(tmp_path):
    h5_file = h5py.File(tmp_path/"foo.hdf", "w")
    extent = (0,10,0,11,0,13)
    origin = (.1,.1,0)
    spacing = (1,2,4)
    direction = (0,0,1,1,0,0,0,1,0)
    ida.init_vtkhdf(h5_file, extent, origin=origin,
                    spacing=spacing, direction=direction)
    assert bool(h5_file["VTKHDF"])
    np.testing.assert_equal(h5_file.get("VTKHDF").attrs["Version"],
                            np.array([1,0]))
    np.testing.assert_equal(h5_file.get("VTKHDF").attrs["Type"],
                            b"ImageData")
    np.testing.assert_equal(h5_file.get("VTKHDF").attrs["Type"],
                            b"ImageData")
    np.testing.assert_equal(h5_file.get("VTKHDF").attrs["WholeExtent"],
                            extent)
    np.testing.assert_equal(h5_file.get("VTKHDF").attrs["Origin"],
                            origin)
    np.testing.assert_equal(h5_file.get("VTKHDF").attrs["Spacing"],
                            spacing)
    np.testing.assert_equal(h5_file.get("VTKHDF").attrs["Direction"],
                            direction)
    assert bool(h5_file["VTKHDF"]["PointData"])
    h5_file.close()

def test_create_dataset(tmp_path):
    h5_file = h5py.File(tmp_path/"foo.hdf", "w")
    dim = (11,23,15)
    extent=(0,10,0,22,0,14)
    ida.init_vtkhdf(h5_file, extent)
    ida.create_dataset(h5_file, dim, "myvar")
    assert h5_file["VTKHDF"]["PointData"].attrs["Scalars"] == b"myvar"
    h5_file.close()

def test_write_slice(tmp_path, radial_box):
    box = radial_box()
    arr = ida.get_var(box, "data")
    with h5py.File(tmp_path/"foo.hdf", "w") as h5_file:
        ida.init_vtkhdf(h5_file, box.extent)
        ida.create_dataset(h5_file, box.dimensions, "newvar")
        for i in range(box.dimensions[2]):
            ida.write_slice(h5_file, arr[:,:,i], "newvar", i)

    with h5py.File(tmp_path/"foo.hdf", "r") as h5_file:
        for i in range(box.dimensions[2]):
            slice = ida.read_slice(h5_file, "newvar", i)
            assert slice.shape == box.dimensions[:-1]
            np.testing.assert_allclose(slice, ida.get_var(box, "data")[:,:,i])