import batoid
from scipy.spatial.transform import Rotation as R
import numpy as np

def get_list_of_optics(telescope):
    optics = list()
    for one in telescope.items:
        [optics.append(two.name) for two in telescope[one.name].items]
    return optics

def make_optics_reflective(telescope, r_frac=0.02):
    for surface in telescope.itemDict.values():
        if isinstance(surface, batoid.RefractiveInterface):
            surface.forwardCoating = batoid.SimpleCoating(r_frac, 1-r_frac)
            surface.reverseCoating = batoid.SimpleCoating(r_frac, 1-r_frac)
        if isinstance(surface, batoid.Detector):
            surface.forwardCoating = batoid.SimpleCoating(r_frac, 1-r_frac)
    return r_frac

# get lens position information from compound optics
def getOptPosition(telescope, name, i):
    return telescope[name].coordSys.origin[i]

def getOptPosX(telescope, name):
    return getOptPosition(telescope, name, 0)

def getOptPosY(telescope, name):
    return getOptPosition(telescope, name, 1)

def getOptPosZ(telescope, name):
    return getOptPosition(telescope, name, 2)

# function to rotate one element of a telescope
def rotate_optic(telescope, name, axis='y', angle=1, verbose=False):
    # Rotating
    rot = R.from_euler(axis, angle, degrees=True)
    if verbose:
        print('Rotation around Y as Euler:\n', rot.as_euler('zyx', degrees=True))
        print('Rotation around Y as  matrix:\n', rot.as_matrix())
    # Rotating one item of the telescope
    rotated_telescope = telescope.withLocallyRotatedOptic(name=name, rot=rot.as_matrix())
    if verbose:
        print(f'{name} before rotation:\n', telescope[name].coordSys.rot)
        print(f'{name} after rotation:\n', rotated_telescope[name].coordSys.rot)
    return rotated_telescope

# function to translate one element of a telescope
def translate_optic(telescope, name, axis='x', distance=0.01, verbose=False):
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
    # Rotating around the 3 axis
    rotX = R.from_euler('x', angles[0], degrees=True)
    rotXY = rotX*R.from_euler('y', angles[1], degrees=True)
    rotXYZ = rotXY*R.from_euler('z', angles[2], degrees=True)
    if verbose:
        print('Rotation around Y as Euler:\n', rotXYZ.as_euler('zyx', degrees=True))
        print('Rotation around Y as  matrix:\n', rotXYZ.as_matrix())
    # Rotating one item of the telescope
    rotated_telescope = telescope.withLocallyRotatedOptic(name=name, rot=rotXYZ.as_matrix())
    if verbose:
        print(f'{name} before rotation:\n', telescope[name].coordSys.rot)
        print(f'{name} after rotation:\n', rotated_telescope[name].coordSys.rot)
    return rotated_telescope

# function to translate one element of a telescope
def translate_optic_vector(telescope, name, shifts=[0.001, 0.001, 0.001], verbose=False):
    translated_telescope = telescope.withLocallyShiftedOptic(name=name, shift=shifts)
    return translated_telescope

def randomized_telescope(telescope, max_angle=0.1, max_shift=0.001, verbose=False):
    # randomly rotate all optical elements
    rnd_telescope = telescope
    optics_names = ['L1', 'L2', 'L3', 'Filter'] #get_list_of_optics(telescope)
    # rotations
    for optic in optics_names:
        rnd_euler_angles = max_angle*(2*np.random.random([3]) - 1)
        rnd_telescope = rotate_optic_vector(rnd_telescope, name=optic, angles=rnd_euler_angles, verbose=verbose)
    #translations
    for optic in optics_names:
        rnd_shifts = max_shift*(2*np.random.random([3]) - 1)
        rnd_telescope = translate_optic_vector(rnd_telescope, name=optic, shifts=rnd_shifts, verbose=verbose)
    return rnd_telescope


if __name__ == '__main__':
    # test list of optics
    assert get_list_of_optics(telescope)==['L1', 'L2', 'Filter', 'L3', 'Detector'], 'Not a CCOB optical setup'
