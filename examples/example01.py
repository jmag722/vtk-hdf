import h5py
import numpy as np
import pyvista

import vtkhdf.image as v5i


def example01():
    dimensions = (91, 51, 121)
    spacing = (.01, .013, .03)
    origin = v5i.origin_of_centered_image(dimensions, spacing, 2)
    box = pyvista.ImageData(
        dimensions=dimensions,
        spacing=spacing,
        origin=origin
    )
    # dataset small enough that we can get away with meshgrid
    X, Y, _ = v5i.mesh_axes(
        *v5i.get_point_axes(box.dimensions, box.spacing, box.origin))
    data = X*X+Y*Y
    v5i.set_point_array(box, data, "data")
    filename = f"myimage{v5i.EXTENSION}"
    with h5py.File(filename, "w") as f:
        v5i.write_vtkhdf(f, box)
    mesh = pyvista.wrap(v5i.read_vtkhdf(filename))
    np.testing.assert_allclose(
        v5i.get_point_array(mesh, "data"),
        v5i.get_point_array(box, "data")
    )


if __name__ == "__main__":
    example01()
