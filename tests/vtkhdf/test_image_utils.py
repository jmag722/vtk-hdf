import numpy as np
import vtkhdf.image_utils as iu

def test_point2cell_dimension():
    assert iu.point2cell_dimension([5]) == (4,)
    assert iu.point2cell_dimension([1]) == (1,)
    assert iu.point2cell_dimension([1,2,3]) == (1,1,2)

def test_point2cell_origin():
    assert iu.point2cell_origin(5, 3, 0.1) == 1.6
    assert iu.point2cell_origin(1, 4, 0) == 0

def test_axis_length():
    d = 5
    s = .25
    assert iu.axis_length(d,s) == 1
    d = (1,2,3)
    s = (0.1, 1, 4)
    actual = [iu.axis_length(d,s) for d,s in zip(d,s)]
    expected = [0., 1., 8.]
    assert actual == expected
    np.testing.assert_allclose(iu.axis_length(np.array(d), np.array(s)),
                               np.array(expected))

def test_origin_of_centered_image():
    dim = (15,16,17)
    spacing = (1,1,3)
    actual = iu.origin_of_centered_image(dim, spacing, True)
    expected = np.array([-7,-7.5,0])
    np.testing.assert_equal(actual, expected)
    actual = iu.origin_of_centered_image(dim, spacing, False)
    expected = np.array([-7,-7.5,-24])
    np.testing.assert_equal(actual, expected)
    actual = iu.origin_of_centered_image(5, 2)
    expected = np.array([-4])
    np.testing.assert_equal(actual, expected)

def test_get_point_axis():
    np.testing.assert_equal(iu.get_point_axis(5,1,0),
                            np.array([0, 1, 2, 3, 4]))
    np.testing.assert_allclose(iu.get_point_axis(4,.1,-.5),
                               np.array([-.5, -.4, -.3, -.2]))
    np.testing.assert_allclose(iu.get_point_axis(1,.1,-.5),
                               np.array([-0.5]))
    
def test_get_cell_axis():
    np.testing.assert_equal(iu.get_cell_axis(5,1,0),
                            np.array([0.5, 1.5, 2.5, 3.5]))
    np.testing.assert_allclose(iu.get_cell_axis(4,.1,-.5),
                               np.array([-.45, -.35, -.25]))
    np.testing.assert_allclose(iu.get_cell_axis(1,.1,-.5),
                               np.array([-0.5]))

def test_get_point_axes():
    dim = (5,4,3)
    s = (1,2,4)
    o = (0, 1, .3)
    x,y,z = iu.get_point_axes(dim, s, o)
    np.testing.assert_allclose(x, np.array([0, 1, 2, 3, 4]))
    np.testing.assert_allclose(y, np.array([1, 3, 5, 7]))
    np.testing.assert_allclose(z, np.array([.3, 4.3, 8.3]))
    
    dim = (5,4,1)
    x,y,z = iu.get_point_axes(dim, s, o)
    np.testing.assert_allclose(z, np.array([.3]))

def test_get_cell_axes():
    dim = (5,4,3)
    s = (1,2,4)
    o = (0, 1, .3)
    x,y,z = iu.get_cell_axes(dim, s, o)
    np.testing.assert_allclose(x, np.array([.5, 1.5, 2.5, 3.5]))
    np.testing.assert_allclose(y, np.array([2, 4, 6]))
    np.testing.assert_allclose(z, np.array([2.3, 6.3]))
    
    dim = (5,4,1)
    x,y,z = iu.get_cell_axes(dim, s, o)
    np.testing.assert_allclose(z, np.array([.3]))
