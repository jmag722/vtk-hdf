import numpy as np
import vtkhdf_wrappers.array_helpers as ah

def test_compute_origin():
    dim = (15,16,17)
    spacing = (1,1,3)
    actual = ah.compute_origin(dim, spacing, True)
    expected = np.array([-7,-7.5,0])
    np.testing.assert_equal(actual, expected)
    actual = ah.compute_origin(dim, spacing, False)
    expected = np.array([-7,-7.5,-24])
    np.testing.assert_equal(actual, expected)
    actual = ah.compute_origin(5, 2)
    expected = np.array([-4])
    np.testing.assert_equal(actual, expected)

def test_compute_axis_array():
    np.testing.assert_equal(ah.compute_axis_array(5,1,0),
                            np.array([0, 1, 2, 3, 4]))
    np.testing.assert_allclose(ah.compute_axis_array(4,.1,-.5),
                               np.array([-.5, -.4, -.3, -.2]))

def test_compute_axis_arrays():
    dim = (5,4,3)
    s = (1,2,4)
    o = (0, 1, .3)
    x,y,z = ah.compute_axis_arrays(dim, s, o)
    np.testing.assert_allclose(x,
                               np.array([0, 1, 2, 3, 4]))
    np.testing.assert_allclose(y,
                               np.array([1, 3, 5, 7]))
    np.testing.assert_allclose(z,
                               np.array([.3, 4.3, 8.3]))