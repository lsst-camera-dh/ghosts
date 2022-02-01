import batoid
from scipy.spatial.transform import Rotation as transform_rotation
import numpy as np


def get_list_of_optics(telescope):
    """ Get a simple list of optical elements from a `batoid` telescope

    Parameters
    ----------
    telescope : `batoid.telescope`
        the optical setup as defined in `batoid`

    Returns
    -------
    optics : `list` of `string` objects
        a simple list of optical element names
    """
    optics = list()
    for one in telescope.items:
        [optics.append(two.name) for two in telescope[one.name].items]
    return optics


def make_optics_reflective(telescope, r_frac=0.02):
    """ Applies a simple coating as a unique refraction index for each optical element surface

    .. todo::
        `make_optics_reflective` should implement wavelength dependent coating for each optical surface

    Parameters
    ----------
    telescope : `batoid.telescope`
        the optical setup as defined in `batoid`
    r_frac : `float`
        a refraction index, usually of the order of 0.02

    Returns
    -------
    r_frac : `float`
        the refraction index applied
    """
    for surface in telescope.itemDict.values():
        if isinstance(surface, batoid.RefractiveInterface):
            surface.forwardCoating = batoid.SimpleCoating(r_frac, 1 - r_frac)
            surface.reverseCoating = batoid.SimpleCoating(r_frac, 1 - r_frac)
        if isinstance(surface, batoid.Detector):
            surface.forwardCoating = batoid.SimpleCoating(r_frac, 1 - r_frac)
    return r_frac


def get_optics_position(telescope, name, axis_i):
    """ Internal interface to get the position of an optical element given its name

    .. todo::
        `get_optics_position` should be a hidden internal interface

    Parameters
    ----------
    telescope : `batoid.telescope`
        the optical setup as defined in `batoid`
    name : `string`
        the name of an optical element
    axis_i : `int`
        the axis index in [0, 1, 2]

    Returns
    -------
    position_i : `float`
        the position along the axis index i
    """
    position_i = telescope[name].coordSys.origin[axis_i]
    return position_i


def get_optics_position_x(telescope, name):
    """ Proxy to get the position of an optical element along the x axis

    Parameters
    ----------
    telescope : `batoid.telescope`
        the optical setup as defined in `batoid`
    name : `string`
        the name of an optical element

    Returns
    -------
    pos_x : `float`
        the position along the axis x
    """
    pos_x = get_optics_position(telescope, name, 0)
    return pos_x


def get_optics_position_y(telescope, name):
    """ Proxy to get the position of an optical element along the y axis

    Parameters
    ----------
    telescope : `batoid.telescope`
        the optical setup as defined in `batoid`
    name : `string`
        the name of an optical element

    Returns
    -------
    pos_y : `float`
        the position along the axis y
    """
    pos_y = get_optics_position(telescope, name, 1)
    return pos_y


def get_optics_position_z(telescope, name):
    """ Proxy to get the position of an optical element along the z axis

    Parameters
    ----------
    telescope : `batoid.telescope`
        the optical setup as defined in `batoid`
    name : `string`
        the name of an optical element

    Returns
    -------
    pos_z : `float`
        the position along the axis z
    """
    pos_z = get_optics_position(telescope, name, 2)
    return pos_z


def rotate_optic(telescope, name, axis='y', angle=1, verbose=False):
    """ Rotate one optical element of a telescope around a given axis and for a given rotation angle

    Parameters
    ----------
    telescope : `batoid.telescope`
        the optical setup as defined in `batoid`
    name : `string`
        the name of an optical element
    axis : `string`
        the name of the rotation axis, usually y
    angle : `float`
        the value of the rotation angle in degrees

    Returns
    -------
    rotated_telescope : `batoid.telescope`
        a new telescope with a rotated optical element
    """
    # Rotating
    rot = transform_rotation.from_euler(axis, angle, degrees=True)
    if verbose:
        print('Rotation around Y as Euler:\n', rot.as_euler('zyx', degrees=True))
        print('Rotation around Y as  matrix:\n', rot.as_matrix())
    # Rotating one item of the telescope
    rotated_telescope = telescope.withLocallyRotatedOptic(name=name, rot=rot.as_matrix())
    if verbose:
        print(f'{name} before rotation:\n', telescope[name].coordSys.rot)
        print(f'{name} after rotation:\n', rotated_telescope[name].coordSys.rot)
    return rotated_telescope


def translate_optic(telescope, name, axis='x', distance=0.01):
    """ Translate one optical element of a telescope along a given axis and for a given length

    Parameters
    ----------
    telescope : `batoid.telescope`
        the optical setup as defined in `batoid`
    name : `string`
        the name of an optical element
    axis : `string`
        the name of the rotation axis, usually y
    distance : `float`
        the value of the shift in meters

    Returns
    -------
    rotated_telescope : `batoid.telescope`
        a new telescope with a rotated optical element
    """
    vector = [0, 0, 0]
    # translating
    if axis == 'x':
        vector = [distance, 0, 0]
    elif axis == 'y':
        vector = [0, distance, 0]
    elif axis == 'z':
        vector = [0, 0, distance]
    translated_telescope = telescope.withLocallyShiftedOptic(name=name, shift=vector)
    return translated_telescope


# function to rotate one element of a telescope
def rotate_optic_vector(telescope, name, angles=[0.1, 0.1, 0.1], verbose=False):
    """ Rotate one optical element of a telescope given a list of Euler angles

    Parameters
    ----------
    telescope : `batoid.telescope`
        the optical setup as defined in `batoid`
    name : `string`
        the name of an optical element
    angles : `list` of `floats`
        the values of Eulers angles in degrees as a list
    verbose : `bool`
        the verbose mode, true or false

    Returns
    -------
    rotated_telescope : `batoid.telescope`
        a new telescope with a rotated optical element
    """
    # Rotating around the 3 axis
    rot_x = transform_rotation.from_euler('x', angles[0], degrees=True)
    rot_xy = rot_x * transform_rotation.from_euler('y', angles[1], degrees=True)
    rot_xyz = rot_xy * transform_rotation.from_euler('z', angles[2], degrees=True)
    if verbose:
        print('Rotation around Y as Euler:\n', rot_xyz.as_euler('zyx', degrees=True))
        print('Rotation around Y as  matrix:\n', rot_xyz.as_matrix())
    # Rotating one item of the telescope
    rotated_telescope = telescope.withLocallyRotatedOptic(name=name, rot=rot_xyz.as_matrix())
    if verbose:
        print(f'{name} before rotation:\n', telescope[name].coordSys.rot)
        print(f'{name} after rotation:\n', rotated_telescope[name].coordSys.rot)
    return rotated_telescope


# function to translate one element of a telescope
def translate_optic_vector(telescope, name, shifts=[0.001, 0.001, 0.001]):
    """ Translate an optical element of a telescope given a list of shifts along x, y and z axis

    Parameters
    ----------
    telescope : `batoid.telescope`
        the optical setup as defined in `batoid`
    name : `string`
        the name of an optical element
    shifts : `list` of `floats`
        the values of shifts in meters as a list for x, y and z axis.

    Returns
    -------
    rotated_telescope : `batoid.telescope`
        a new telescope with a rotated optical element
    """
    translated_telescope = telescope.withLocallyShiftedOptic(name=name, shift=shifts)
    return translated_telescope


def randomized_telescope(telescope, max_angle=0.1, max_shift=0.001, verbose=False):
    """ Randomly translates and rotates all optical elements of a telescope
    according to uniform distributions drown from the given a maximum rotation angle
    and shift.

    Rotation angles are drown from a uniform distribution in [-max_angle; +max_angle]

    Translation values are drown from a uniform distribution in [-max_shift; +max_shift]

    Parameters
    ----------
    telescope : `batoid.telescope`
        the optical setup as defined in `batoid`
    max_angle : `float`
        the maximum value of the rotation angle in degree
    max_shift : `floats`
        the maximum value of the shift in meters
    verbose : `bool`
        the verbose mode, true or false

    Returns
    -------
    rnd_telescope : `batoid.telescope`
        a new telescope with randomized optics position and rotation
    """
    # randomly rotate all optical elements
    rnd_telescope = telescope
    optics_names = ['L1', 'L2', 'L3', 'Filter']  # get_list_of_optics(telescope)
    # rotations
    for optic in optics_names:
        rnd_euler_angles = max_angle * 2 * (np.random.random([3]) - 0.5)
        rnd_telescope = rotate_optic_vector(rnd_telescope, name=optic, angles=rnd_euler_angles, verbose=verbose)
    # translations
    for optic in optics_names:
        rnd_shifts = max_shift * 2 * (np.random.random([3]) - 0.5)
        rnd_telescope = translate_optic_vector(rnd_telescope, name=optic, shifts=rnd_shifts)
    return rnd_telescope


if __name__ == '__main__':
    # test list of optics
    ccob_telescope = batoid.Optic.fromYaml("LSST_CCOB_r.yaml")
    assert get_list_of_optics(ccob_telescope) == ['L1', 'L2', 'Filter', 'L3', 'Detector'], 'Not a CCOB optical setup'
