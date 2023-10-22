import pytest
import numpy as np
import pyvista
import h5py
import vtkhdf.image as v5i

@pytest.fixture
def radial_box():
    def _method():
        dimensions = np.array([151, 91, 113])
        spacing = np.array([.01, .011, .03])
        origin = v5i.origin_of_centered_image(dimensions, spacing)
        box = pyvista.ImageData(
            dimensions=dimensions,
            spacing=spacing,
            origin=origin
        )
        X,Y,_ = v5i.mesh_axes(*v5i.get_axes(box.dimensions, box.spacing,
                                                box.origin))
        data = np.sqrt(X*X+Y*Y)
        v5i.set_array(box, data, "data")
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
        np.testing.assert_allclose(slice, v5i.get_array(box, "data")[:,:,i])
    h5_file.close()

def test_read_vtkhdf(tmp_path, radial_box):
    box = radial_box()
    v5i.write_vtkhdf(tmp_path/"mybox-vti.hdf", box)
    readin = v5i.read_vtkhdf(tmp_path/"mybox-vti.hdf")
    np.testing.assert_allclose(v5i.get_array(box, "data"),
                               v5i.get_array(readin, "data"))

def test_initialize(tmp_path):
    h5_file = h5py.File(tmp_path/"foo.hdf", "w")
    extent = (0,10,0,11,0,13)
    origin = (.1,.1,0)
    spacing = (1,2,4)
    direction = (0,0,1,1,0,0,0,1,0)
    v5i.initialize(h5_file, extent, origin=origin,
                    spacing=spacing, direction=direction)
    assert bool(h5_file[v5i.VTKHDF])
    np.testing.assert_equal(h5_file.get(v5i.VTKHDF).attrs[v5i.VERSION],
                            np.array([1,0]))
    np.testing.assert_equal(h5_file.get(v5i.VTKHDF).attrs[v5i.TYPE],
                            np.string_(v5i.IMAGEDATA))
    np.testing.assert_equal(h5_file.get(v5i.VTKHDF).attrs[v5i.EXTENT],
                            extent)
    np.testing.assert_equal(h5_file.get(v5i.VTKHDF).attrs[v5i.ORIGIN],
                            origin)
    np.testing.assert_equal(h5_file.get(v5i.VTKHDF).attrs[v5i.SPACING],
                            spacing)
    np.testing.assert_equal(h5_file.get(v5i.VTKHDF).attrs[v5i.DIRECTION],
                            direction)
    assert bool(h5_file[v5i.VTKHDF][v5i.POINTDATA])
    h5_file.close()

def test_create_dataset(tmp_path):
    h5_file = h5py.File(tmp_path/"foo.hdf", "w")
    dim = (11,23,15)
    v5i.initialize(h5_file, v5i.dimensions2extent(dim))
    v5i.create_dataset(h5_file, "myvar")
    assert h5_file[v5i.VTKHDF][v5i.POINTDATA].attrs[v5i.SCALARS] == b"myvar"
    assert h5_file[v5i.VTKHDF][v5i.POINTDATA]["myvar"].shape == (15,23,11)
    assert h5_file[v5i.VTKHDF][v5i.POINTDATA]["myvar"].chunks == (1,23,11)
    h5_file.close()

def test_create_dataset_c(tmp_path):
    h5_file = h5py.File(tmp_path/"foo_c.hdf", "w")
    dim_c = (11,23,15)
    v5i.initialize(h5_file, v5i.dimensions2extent(dim_c[::-1]))
    v5i.create_dataset(h5_file, "myvar")
    assert h5_file[v5i.VTKHDF][v5i.POINTDATA].attrs[v5i.SCALARS] == b"myvar"
    assert h5_file[v5i.VTKHDF][v5i.POINTDATA]["myvar"].shape == (11,23,15)
    assert h5_file[v5i.VTKHDF][v5i.POINTDATA]["myvar"].chunks == (1,23,15)
    h5_file.close()

def test_write_slice(tmp_path, radial_box):
    box = radial_box()
    arr = v5i.get_array(box, "data")
    with h5py.File(tmp_path/"foo.hdf", "w") as h5_file:
        v5i.initialize(h5_file, box.extent)
        v5i.create_dataset(h5_file, "newvar")
        for i in range(box.dimensions[2]):
            v5i.write_slice(h5_file, arr[:,:,i], "newvar", i)

    with h5py.File(tmp_path/"foo.hdf", "r") as h5_file:
        for i in range(box.dimensions[2]):
            slice = v5i.read_slice(h5_file, "newvar", i)
            assert slice.shape == box.dimensions[:-1]
            np.testing.assert_allclose(slice, v5i.get_array(box, "data")[:,:,i])

def test_write_slice_c(tmp_path):
    shape_c = (1,10,4)
    arr = np.random.rand(*shape_c)
    file = "foo_c.hdf"
    with h5py.File(tmp_path/file, "w") as h5_file:
        v5i.initialize(h5_file, v5i.dimensions2extent(shape_c[::-1]))
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
    assert v5i.extent2dimensions((0,5,0,3,0,4)) == (6,4,5)
    assert v5i.extent2dimensions((0,15,0,1,0,0,0,3)) == (16,2,1,4)

def test_origin_of_centered_image():
    dim = (15,16,17)
    spacing = (1,1,3)
    actual = v5i.origin_of_centered_image(dim, spacing, True)
    expected = np.array([-7,-7.5,0])
    np.testing.assert_equal(actual, expected)
    actual = v5i.origin_of_centered_image(dim, spacing, False)
    expected = np.array([-7,-7.5,-24])
    np.testing.assert_equal(actual, expected)
    actual = v5i.origin_of_centered_image(5, 2)
    expected = np.array([-4])
    np.testing.assert_equal(actual, expected)

def test_get_axis():
    np.testing.assert_equal(v5i.get_axis(5,1,0),
                            np.array([0, 1, 2, 3, 4]))
    np.testing.assert_allclose(v5i.get_axis(4,.1,-.5),
                               np.array([-.5, -.4, -.3, -.2]))

def test_get_axes():
    dim = (5,4,3)
    s = (1,2,4)
    o = (0, 1, .3)
    x,y,z = v5i.get_axes(dim, s, o)
    np.testing.assert_allclose(x,
                               np.array([0, 1, 2, 3, 4]))
    np.testing.assert_allclose(y,
                               np.array([1, 3, 5, 7]))
    np.testing.assert_allclose(z,
                               np.array([.3, 4.3, 8.3]))
