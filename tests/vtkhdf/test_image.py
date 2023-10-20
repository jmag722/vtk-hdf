import pytest
import numpy as np
import pyvista
import h5py
import vtkhdf.utils as ut
import vtkhdf.image as v5i

@pytest.fixture
def radial_box():
    def _method():
        dimensions = np.array([151, 91, 113])
        spacing = np.array([.01, .011, .03])
        origin = ut.compute_origin(dimensions, spacing)
        box = pyvista.ImageData(
            dimensions=dimensions,
            spacing=spacing,
            origin=origin
        )
        X,Y,_ = ut.mesh(*ut.compute_axis_arrays(box.dimensions, box.spacing,
                                                box.origin))
        data = np.sqrt(X*X+Y*Y)
        v5i.set_imagedata(box, data, "data")
        return box
    return _method

def test_read_slice(tmp_path, radial_box):
    box = radial_box()
    v5i.write_vtkhdf(tmp_path/"mybox-vti.hdf", box)
    h5_file = h5py.File(tmp_path/"mybox-vti.hdf", "r")
    for i in range(box.dimensions[2]):
        slice = v5i.read_slice(h5_file, "data", i)
        assert slice.flags.f_contiguous
        assert slice.shape == box.dimensions[:-1]
        np.testing.assert_allclose(slice, v5i.get_imagedata(box, "data")[:,:,i])
    h5_file.close()

def test_read_vtkhdf(tmp_path, radial_box):
    box = radial_box()
    v5i.write_vtkhdf(tmp_path/"mybox-vti.hdf", box)
    readin = v5i.read_vtkhdf(tmp_path/"mybox-vti.hdf")
    np.testing.assert_allclose(v5i.get_imagedata(box, "data"),
                               v5i.get_imagedata(readin, "data"))

def test_init_vtkhdf(tmp_path):
    h5_file = h5py.File(tmp_path/"foo.hdf", "w")
    extent = (0,10,0,11,0,13)
    origin = (.1,.1,0)
    spacing = (1,2,4)
    direction = (0,0,1,1,0,0,0,1,0)
    v5i.init_vtkhdf(h5_file, extent, origin=origin,
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
    v5i.init_vtkhdf(h5_file, v5i.dimensions2extent(dim))
    v5i.create_dataset(h5_file, "myvar")
    assert h5_file["VTKHDF"]["PointData"].attrs["Scalars"] == b"myvar"
    assert h5_file["VTKHDF"]["PointData"]["myvar"].shape == (15,23,11)
    assert h5_file["VTKHDF"]["PointData"]["myvar"].chunks == (1,23,11)
    h5_file.close()

def test_create_dataset_c(tmp_path):
    h5_file = h5py.File(tmp_path/"foo_c.hdf", "w")
    dim_c = (11,23,15)
    v5i.init_vtkhdf(h5_file, v5i.dimensions2extent(dim_c[::-1]))
    v5i.create_dataset(h5_file, "myvar")
    assert h5_file["VTKHDF"]["PointData"].attrs["Scalars"] == b"myvar"
    assert h5_file["VTKHDF"]["PointData"]["myvar"].shape == (11,23,15)
    assert h5_file["VTKHDF"]["PointData"]["myvar"].chunks == (1,23,15)
    h5_file.close()

def test_write_slice(tmp_path, radial_box):
    box = radial_box()
    arr = v5i.get_imagedata(box, "data")
    with h5py.File(tmp_path/"foo.hdf", "w") as h5_file:
        v5i.init_vtkhdf(h5_file, box.extent)
        v5i.create_dataset(h5_file, "newvar")
        for i in range(box.dimensions[2]):
            v5i.write_slice(h5_file, arr[:,:,i], "newvar", i)

    with h5py.File(tmp_path/"foo.hdf", "r") as h5_file:
        for i in range(box.dimensions[2]):
            slice = v5i.read_slice(h5_file, "newvar", i)
            assert slice.shape == box.dimensions[:-1]
            np.testing.assert_allclose(slice, v5i.get_imagedata(box, "data")[:,:,i])

def test_write_slice_c(tmp_path):
    shape_c = (1,10,4)
    arr = np.random.rand(*shape_c)
    file = "foo_c.hdf"
    with h5py.File(tmp_path/file, "w") as h5_file:
        v5i.init_vtkhdf(h5_file, v5i.dimensions2extent(shape_c[::-1]))
        v5i.create_dataset(h5_file, "newvar")
        v5i.write_slice(h5_file, arr[0,:,:], "newvar", 0)

    with h5py.File(tmp_path/file, "r") as h5_file:
        slice = v5i.read_slice(h5_file, "newvar", 0)
        assert slice.shape == shape_c[::-1][:-1]
        np.testing.assert_allclose(slice, v5i.c2f_reshape(arr[0,:,:]))

def test_dimensions2extent():
    assert v5i.dimensions2extent((1,2,3)) == (0,0,0,1,0,2)
    assert v5i.dimensions2extent((5,3,1,2)) == (0,4,0,2,0,0,0,1)

def test_extent2dimensions():
    assert v5i.extent2dimension((0,5,0,3,0,4)) == (6,4,5)
    assert v5i.extent2dimension((0,15,0,1,0,0,0,3)) == (16,2,1,4)
