import pytest
import numpy as np
import pyvista
import h5py
import vtkhdf.image as v5i

PVAR1 = "pointvar1" # dummy point variable
PVAR2 = "pointvar2"
CVAR1 = "cellvar1" # dummy cell variable
FVAR1 = "fieldvar1" # dummy field variable
FVAR2 = "fieldvar2"
DUMMY_IMAGE = "mybox-vti.hdf"

@pytest.fixture
def radial_box():
    def _method():
        dimensions = np.array([151, 91, 113])
        spacing = np.array([.01, .011, .03])
        origin = v5i.origin_of_centered_image(dimensions, spacing, 2)
        box = pyvista.ImageData(
            dimensions=dimensions,
            spacing=spacing,
            origin=origin
        )
        X,Y,_ = v5i.mesh_axes(*v5i.get_point_axes(box.dimensions, box.spacing,
                                                box.origin))
        data = np.sqrt(X*X+Y*Y)
        data2 = X+45
        pts_cc = box.cell_centers().points
        xc = pts_cc[:,0]
        yc = pts_cc[:,1]
        zc = pts_cc[:,2]
        data_c = np.sqrt(xc*xc+yc*yc)
        v5i.set_point_array(box, data, PVAR1)
        v5i.set_point_array(box, data2, PVAR2)
        box.cell_data[CVAR1] = data_c
        box.field_data[FVAR1] = [129]
        box.field_data[FVAR2] = ["/path/to/01-03-2023/directory"]
        return box
    return _method

@pytest.fixture
def write_dummy_image(tmp_path, radial_box):
    def _method():
        box = radial_box()
        with h5py.File(tmp_path/DUMMY_IMAGE, "w") as f:
            v5i.write_vtkhdf(f, box)
        return box
    return _method

def test_read_vtkhdf(tmp_path, write_dummy_image):
    box = write_dummy_image()
    readin = pyvista.wrap(v5i.read_vtkhdf(tmp_path/DUMMY_IMAGE))
    for var in box.point_data: # pyvista doesn't support vtkCellData read?
        np.testing.assert_allclose(v5i.get_point_array(box, var),
                                   v5i.get_point_array(readin, var))

def test_get_dataset(tmp_path, write_dummy_image):
    box = write_dummy_image()
    with h5py.File(tmp_path/DUMMY_IMAGE, "r") as f:
        dset = v5i.c2f_reshape(v5i.get_dataset(f, PVAR1))
        np.testing.assert_allclose(dset, v5i.get_point_array(box, PVAR1))
        dset = v5i.c2f_reshape(v5i.get_dataset(f, CVAR1))
        np.testing.assert_allclose(dset, v5i.get_cell_array(box, CVAR1))

def test_read_slice(tmp_path, write_dummy_image):
    box = write_dummy_image()
    with h5py.File(tmp_path/DUMMY_IMAGE, "r") as f:
        for var in box.array_names:
            dset = v5i.get_dataset(f, var)
            if var in box.point_data:
                arr = v5i.get_point_array(box, var)
                shape = box.dimensions
            elif var in box.cell_data:
                arr = v5i.get_cell_array(box, var)
                shape = v5i.point2cell_dimensions(box.dimensions)
            elif var in box.field_data:
                continue
            for i in range(shape[2]):
                slice = v5i.read_slice(dset, i)
                assert slice.flags.f_contiguous
                assert not slice.flags.c_contiguous
                assert slice.shape == shape[:-1]
                np.testing.assert_allclose(slice, arr[:,:,i])

def test_read_slice_c(tmp_path, write_dummy_image):
    box = write_dummy_image()
    with h5py.File(tmp_path/DUMMY_IMAGE, "r") as f:
        for var in box.array_names:
            dset = v5i.get_dataset(f, var)
            if var in box.point_data:
                arr = v5i.get_point_array(box, var)
                shape = box.dimensions
            elif var in box.cell_data:
                arr = v5i.get_cell_array(box, var)
                shape = v5i.point2cell_dimensions(box.dimensions)
            elif var in box.field_data:
                continue
            arr = v5i.f2c_reshape(arr)
            for i in range(shape[2]):
                slice = v5i.read_slice(dset, i, False)
                assert slice.flags.c_contiguous
                assert not slice.flags.f_contiguous
                assert slice.shape == shape[:-1][::-1]
                np.testing.assert_allclose(
                    slice,  arr[i,:,:]
                )

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
    assert bool(h5_file[v5i.VTKHDF][v5i.CELLDATA])
    assert bool(h5_file[v5i.VTKHDF][v5i.FIELDDATA])
    h5_file.close()

def test_create_field_dataset(tmp_path):
    h5_file = h5py.File(tmp_path/"foo.hdf", "w")
    dim = (11,23,15)
    v5i.initialize(h5_file, v5i.dimensions2extent(dim))
    v5i.create_field_dataset(h5_file, var=FVAR1, data=[-131.55])
    v5i.create_field_dataset(h5_file, var=FVAR2, data=[b"hey you"])
    v5i.create_field_dataset(h5_file, var="nothing", shape=(3,))
    assert bool(h5_file[v5i.VTKHDF][v5i.FIELDDATA][FVAR1])
    assert bool(h5_file[v5i.VTKHDF][v5i.FIELDDATA][FVAR2])
    assert bool(h5_file[v5i.VTKHDF][v5i.FIELDDATA]["nothing"])
    h5_file.close()

def test_get_field_dataset(tmp_path, write_dummy_image):
    box = write_dummy_image()
    with h5py.File(tmp_path/DUMMY_IMAGE, "r") as f:
        for var in box.field_data.keys():
            dset = v5i.get_field_dataset(f, var)
            if isinstance(dset[0], bytes):
                for i,v in enumerate(dset):
                    assert v.decode() == box.field_data[var][i]
            else:
                np.testing.assert_equal(dset, box.field_data[var])

def test_create_point_dataset(tmp_path):
    h5_file = h5py.File(tmp_path/"foo.hdf", "w")
    dim = (11,23,15)
    v5i.initialize(h5_file, v5i.dimensions2extent(dim))
    v5i.create_point_dataset(h5_file, PVAR1, compression="lzf")
    assert bool(h5_file[v5i.VTKHDF][v5i.POINTDATA][PVAR1])
    assert h5_file[v5i.VTKHDF][v5i.POINTDATA].attrs[v5i.SCALARS] == np.string_(PVAR1)
    assert h5_file[v5i.VTKHDF][v5i.POINTDATA][PVAR1].shape == (15,23,11)
    assert h5_file[v5i.VTKHDF][v5i.POINTDATA][PVAR1].chunks == (1,23,11)
    h5_file.close()

def test_create_cell_dataset(tmp_path):
    h5_file = h5py.File(tmp_path/"foo.hdf", "w")
    dim = (11,23,15)
    v5i.initialize(h5_file, v5i.dimensions2extent(dim))
    v5i.create_cell_dataset(h5_file, CVAR1, compression="lzf")
    assert bool(h5_file[v5i.VTKHDF][v5i.CELLDATA][CVAR1])
    assert h5_file[v5i.VTKHDF][v5i.CELLDATA].attrs[v5i.SCALARS] == np.string_(CVAR1)
    assert h5_file[v5i.VTKHDF][v5i.CELLDATA][CVAR1].shape == (14,22,10)
    assert h5_file[v5i.VTKHDF][v5i.CELLDATA][CVAR1].chunks == (1,22,10)

def test_create_point_dataset_c(tmp_path):
    h5_file = h5py.File(tmp_path/"foo_c.hdf", "w")
    dim_c = (11,23,15)
    v5i.initialize(h5_file, v5i.dimensions2extent(dim_c[::-1]))
    v5i.create_point_dataset(h5_file, PVAR1)
    assert bool(h5_file[v5i.VTKHDF][v5i.POINTDATA][PVAR1])
    assert h5_file[v5i.VTKHDF][v5i.POINTDATA].attrs[v5i.SCALARS] == np.string_(PVAR1)
    assert h5_file[v5i.VTKHDF][v5i.POINTDATA][PVAR1].shape == dim_c
    assert h5_file[v5i.VTKHDF][v5i.POINTDATA][PVAR1].chunks == (1,23,15)
    h5_file.close()

def test_get_chunk_shape():
    assert v5i.get_chunk_shape((1,15,25)) == (1,15,25)
    assert v5i.get_chunk_shape((4,11,15)) == (1,11,15)

def test_write_slice(tmp_path, radial_box):
    box = radial_box()
    arr = v5i.get_point_array(box, PVAR1)
    with h5py.File(tmp_path/DUMMY_IMAGE, "w") as h5_file:
        v5i.initialize(h5_file, box.extent)
        dset = v5i.create_point_dataset(h5_file, PVAR1, compression="lzf")
        for i in range(box.dimensions[2]):
            v5i.write_slice(dset, arr[:,:,i], i)

    with h5py.File(tmp_path/DUMMY_IMAGE, "r") as h5_file:
        dset = v5i.get_dataset(h5_file, PVAR1)
        for i in range(box.dimensions[2]):
            slice = v5i.read_slice(dset, i)
            assert slice.shape == box.dimensions[:-1]
            np.testing.assert_allclose(slice, v5i.get_point_array(box, PVAR1)[:,:,i])

def test_write_slice_c(tmp_path):
    shape_c = (1,10,4)
    arr = np.random.rand(*shape_c)
    with h5py.File(tmp_path/DUMMY_IMAGE, "w") as h5_file:
        v5i.initialize(h5_file, v5i.dimensions2extent(shape_c[::-1]))
        dset = v5i.create_point_dataset(h5_file, PVAR1)
        v5i.write_slice(dset, arr[0,:,:], 0)

    with h5py.File(tmp_path/DUMMY_IMAGE, "r") as h5_file:
        dset = v5i.get_dataset(h5_file, PVAR1)
        slice_f = v5i.read_slice(dset, 0)
        assert slice_f.shape == shape_c[::-1][:-1]
        np.testing.assert_allclose(slice_f, v5i.c2f_reshape(arr[0,:,:]))

        slice = v5i.read_slice(dset, 0, False)
        assert slice.shape == shape_c[1:]
        np.testing.assert_allclose(slice, arr[0,:,:])

def test_write_cell_slice(tmp_path, radial_box):
    box = radial_box()
    arr = v5i.get_cell_array(box, CVAR1)
    arr2 = v5i.get_point_array(box, PVAR2)
    cc_shape = v5i.point2cell_dimensions(box.dimensions)
    box.save(tmp_path/"hey.vti")
    with h5py.File(tmp_path/DUMMY_IMAGE, "w") as h5_file:
        v5i.initialize(h5_file, box.extent)
        dset = v5i.create_cell_dataset(h5_file, CVAR1)
        for i in range(cc_shape[2]):
            v5i.write_slice(dset, arr[:,:,i], i)

        dset2 = v5i.create_point_dataset(h5_file, PVAR2)
        for i in range(box.dimensions[2]):
            v5i.write_slice(dset2, arr2[:,:,i], i)
        

    with h5py.File(tmp_path/DUMMY_IMAGE, "r") as h5_file:
        dset = v5i.get_dataset(h5_file, CVAR1)
        dim_cell = v5i.get_cell_data_shape(h5_file)[::-1]
        for i in range(dim_cell[2]):
            slice = v5i.read_slice(dset, i)
            assert slice.shape == dim_cell[:-1]
            np.testing.assert_allclose(slice, v5i.get_cell_array(box, CVAR1)[:,:,i])

def test_get_field_array(radial_box):
    box = radial_box()
    fdat1 = v5i.get_field_array(box, FVAR1)
    fdat2 = v5i.get_field_array(box, FVAR2)
    np.testing.assert_equal(fdat1, box.field_data[FVAR1])
    assert type(fdat2[0]) == bytes
    assert fdat2 == [box.field_data[FVAR2][0].encode()]

def test_dimensions2extent():
    assert v5i.dimensions2extent((1,2,3)) == (0,0,0,1,0,2)
    assert v5i.dimensions2extent((5,3,1,2)) == (0,4,0,2,0,0,0,1)

def test_extent2dimensions():
    assert v5i.extent2dimensions((0,5,1,3,2,4)) == (6, 3, 3)
    assert v5i.extent2dimensions((0,5,0,3,0,4)) == (6,4,5)
    assert v5i.extent2dimensions((0,15,0,1,0,0,0,3)) == (16,2,1,4)

def test_point2cell_dimensions():
    assert v5i.point2cell_dimensions([5]) == (4,)
    assert v5i.point2cell_dimensions([1]) == (1,)
    assert v5i.point2cell_dimensions([1,2,3]) == (1,1,2)

def test_point2cell_extent():
    assert v5i.point2cell_extent([0,1,0,0,1,6]) == (0,0,0,0,1,5)
    assert v5i.point2cell_extent([0,1,0,11,0,0]) == (0,0,0,10,0,0)
    assert v5i.point2cell_extent((1,3,2,11,0,0)) == (1,2,2,10,0,0)

def test_point2cell_origin():
    assert v5i.point2cell_origin((5,), (3,), (0.1,)) == (1.6,)
    assert v5i.point2cell_origin((1,), (4,), (0,)) == (0,)
    assert v5i.point2cell_origin((15,1,7), (1,1,.3), (0,1,0)) == (0.5, 1, 0.15)

def test_axis_length():
    d = 5
    s = .25
    assert v5i.axis_length(d,s) == 1
    d = (1,2,3)
    s = (0.1, 1, 4)
    actual = [v5i.axis_length(d,s) for d,s in zip(d,s)]
    expected = [0., 1., 8.]
    assert actual == expected
    np.testing.assert_allclose(v5i.axis_length(np.array(d), np.array(s)),
                               np.array(expected))

def test_origin_of_centered_image():
    dim = (15,16,17)
    spacing = (1,1,3)
    actual = v5i.origin_of_centered_image(dim, spacing, 2)
    assert actual == (-7,-7.5,0)
    actual = v5i.origin_of_centered_image(dim, spacing, 0)
    assert actual == (0,-7.5,-24)
    actual = v5i.origin_of_centered_image(dim, spacing)
    assert actual == (-7,-7.5,-24)
    actual = v5i.origin_of_centered_image((5,), (2,))
    assert actual == (-4,)
    actual = v5i.origin_of_centered_image((5,), (2,), 0)
    assert actual == (0,)
    with pytest.raises(KeyError):
        v5i.origin_of_centered_image((3,5), (2,4), 2)

def test_get_point_axis():
    np.testing.assert_equal(v5i.get_point_axis(5,1,0),
                            np.array([0, 1, 2, 3, 4]))
    np.testing.assert_allclose(v5i.get_point_axis(4,.1,-.5),
                               np.array([-.5, -.4, -.3, -.2]))
    np.testing.assert_allclose(v5i.get_point_axis(1,.1,-.5),
                               np.array([-0.5]))
    
def test_get_cell_axis():
    np.testing.assert_equal(v5i.get_cell_axis(5,1,0),
                            np.array([0.5, 1.5, 2.5, 3.5]))
    np.testing.assert_allclose(v5i.get_cell_axis(4,.1,-.5),
                               np.array([-.45, -.35, -.25]))
    np.testing.assert_allclose(v5i.get_cell_axis(1,.1,-.5),
                               np.array([-0.5]))

def test_get_point_axes():
    dim = (5,4,3)
    s = (1,2,4)
    o = (0, 1, .3)
    x,y,z = v5i.get_point_axes(dim, s, o)
    np.testing.assert_allclose(x, np.array([0, 1, 2, 3, 4]))
    np.testing.assert_allclose(y, np.array([1, 3, 5, 7]))
    np.testing.assert_allclose(z, np.array([.3, 4.3, 8.3]))
    
    dim = (5,4,1)
    x,y,z = v5i.get_point_axes(dim, s, o)
    np.testing.assert_allclose(z, np.array([.3]))

def test_get_cell_axes():
    dim = (5,4,3)
    s = (1,2,4)
    o = (0, 1, .3)
    x,y,z = v5i.get_cell_axes(dim, s, o)
    np.testing.assert_allclose(x, np.array([.5, 1.5, 2.5, 3.5]))
    np.testing.assert_allclose(y, np.array([2, 4, 6]))
    np.testing.assert_allclose(z, np.array([2.3, 6.3]))
    
    dim = (5,4,1)
    x,y,z = v5i.get_cell_axes(dim, s, o)
    np.testing.assert_allclose(z, np.array([.3]))