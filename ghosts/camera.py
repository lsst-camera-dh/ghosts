"""camera module

This module provides functions to display a dummy camera, with rafts, CCDs and amplifiers.
"""

import numpy as np
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection
from ghosts.constants import LSST_CAMERA_AMP_DX, LSST_CAMERA_AMP_DY, LSST_CAMERA_CCD_DX, LSST_CAMERA_RAFT_DX


def make_amp(x, y):
    """ Build an amplifier rectangle

    Parameters
    ----------
    x : `float`
        x position of the lower left angle of the amplifier rectangle
    y : `float`
        y position of the lower left angle of the amplifier rectangle

    Returns
    -------
    - : `matplotlib.patches.Rectangle`
       a `Rectangle` object
    """
    return Rectangle((x, y), LSST_CAMERA_AMP_DX, LSST_CAMERA_AMP_DY)


def make_ccd(x, y):
    """ Build a CCD rectangle

    Parameters
    ----------
    x : `float`
        x position of the lower left angle of the CCD rectangle
    y : `float`
        y position of the lower left angle of the CCD rectangle

    Returns
    -------
    - : `matplotlib.patches.Rectangle`
       a `Rectangle` object
    """
    return Rectangle((x, y), LSST_CAMERA_CCD_DX, LSST_CAMERA_CCD_DX)


def make_raft(x, y):
    """ Build a raft rectangle

    Parameters
    ----------
    x : `float`
        x position of the lower left angle of the raft rectangle
    y : `float`
        y position of the lower left angle of the raft rectangle

    Returns
    -------
    - : `matplotlib.patches.Rectangle`
       a `Rectangle` object
    """
    return Rectangle((x, y), LSST_CAMERA_RAFT_DX, LSST_CAMERA_RAFT_DX)


def make_ccd_with_amp(x, y):
    """ Build a CCD rectangle filled with Amplifiers
    Parameters
    ----------
    x : `float`
        x position of the lower left angle of the raft rectangle
    y : `float`
        y position of the lower left angle of the raft rectangle

    Returns
    -------
    ccd : `list` of `matplotlib.patches.Rectangle`
       a list of CCDs with amplifiers
    """
    ccd = []
    amp_x = np.arange(x, x+LSST_CAMERA_CCD_DX*0.99, LSST_CAMERA_AMP_DX)
    amp_y = np.arange(y, y+LSST_CAMERA_CCD_DX*0.99, LSST_CAMERA_AMP_DY)
    for rx in amp_x:
        for ry in amp_y:
            ccd.append(make_amp(rx, ry))
    return ccd


def make_raft_with_ccd(x, y):
    """ Build a raft filled with CCDs

    Parameters
    ----------
    x : `float`
        x position of the lower left angle of the raft rectangle
    y : `float`
        y position of the lower left angle of the raft rectangle

    Returns
    -------
    raft : `list` of `matplotlib.patches.Rectangle`
       a list of rafts with CCDs
    """
    raft = []
    ccd_x = np.arange(x, x+LSST_CAMERA_RAFT_DX*0.99, LSST_CAMERA_CCD_DX)
    ccd_y = np.arange(y, y+LSST_CAMERA_RAFT_DX*0.99, LSST_CAMERA_CCD_DX)
    for rx in ccd_x:
        for ry in ccd_y:
            raft.append(make_ccd(rx, ry))
    return raft


def make_raft_with_ccd_with_amp(x, y):
    """ Build a raft filled with CCDs filled with amps

    Parameters
    ----------
    x : `float`
        x position of the lower left angle of the raft rectangle
    y : `float`
        y position of the lower left angle of the raft rectangle

    Returns
    -------
    raft : `list` of `matplotlib.patches.Rectangle`
       a list of CCDs with amplifiers
    """
    raft = []
    ccd_x = np.arange(x, x+LSST_CAMERA_RAFT_DX*0.99, LSST_CAMERA_CCD_DX)
    ccd_y = np.arange(y, y+LSST_CAMERA_RAFT_DX*0.99, LSST_CAMERA_CCD_DX)
    for rx in ccd_x:
        for ry in ccd_y:
            raft.extend(make_ccd_with_amp(rx, ry))
    return raft


def make_raft_with_one_ccd_with_amp(x, y, i_ccd):
    """ Build a raft with one CCD filled with amps

    Parameters
    ----------
    x : `float`
        x position of the lower left angle of the raft rectangle
    y : `float`
        y position of the lower left angle of the raft rectangle
    i_ccd : `int`
        number of the CCD that you wish to fill with amplifiers

    Returns
    -------
    raft : `list` of `matplotlib.patches.Rectangle`
       a list of CCDs and some amplifiers
    """
    raft = []
    ccd_x = np.arange(x, x+LSST_CAMERA_RAFT_DX*0.99, LSST_CAMERA_CCD_DX)
    ccd_y = np.arange(y, y+LSST_CAMERA_RAFT_DX*0.99, LSST_CAMERA_CCD_DX)
    i = 0
    for rx in ccd_x:
        for ry in ccd_y:
            if i == i_ccd:
                raft.extend(make_ccd_with_amp(rx, ry))
            else:
                raft.append(make_ccd(rx, ry))
            i = i+1
    return raft


def build_camera():
    """ Build a camera as a collection of Rectangles

    Parameters
    ----------

    Returns
    -------
    rafts : `matplotlib.collections.PatchCollection`
            a collection of rectangles that looks like the rafts of the Rubin LSST Camera
    ccds : `matplotlib.collections.PatchCollection`
            a collection of rectangles that looks like the CCDs of the Rubin LSST Camera
    amps : `matplotlib.collections.PatchCollection`
            a collection of rectangles that looks like the Amplifiers of the Rubin LSST Camera
    """
    # Rafts range with no gaps
    raft_x = np.arange(-0.325, 0.325, LSST_CAMERA_RAFT_DX)
    raft_y = np.arange(-0.325, 0.325, LSST_CAMERA_RAFT_DX)

    # List of rectangles
    rafts = []
    ccds = []
    amps = []

    # line 1
    for rx in raft_x[1:4]:
        rafts.append(make_raft(rx, raft_y[0]))
        ccds.extend(make_raft_with_ccd(rx, raft_y[0]))
    # line 2
    for rx in raft_x:
        rafts.append(make_raft(rx, raft_y[1]))
        ccds.extend(make_raft_with_ccd(rx, raft_y[1]))
    # line 3
    for i, rx in enumerate(raft_x):
        if i == 1:
            amps.extend(make_raft_with_ccd_with_amp(rx, raft_y[2]))
        elif i == 3:
            amps.extend(make_raft_with_one_ccd_with_amp(rx, raft_y[2], 4))
        else:
            rafts.append(make_raft(rx, raft_y[2]))
            ccds.extend(make_raft_with_ccd(rx, raft_y[2]))
    # line 4
    for rx in raft_x:
        rafts.append(make_raft(rx, raft_y[3]))
        ccds.extend(make_raft_with_ccd(rx, raft_y[3]))
    # line 5
    for rx in raft_x[1:4]:
        rafts.append(make_raft(rx, raft_y[4]))
        ccds.extend(make_raft_with_ccd(rx, raft_y[4]))

    rafts_col = PatchCollection(rafts, facecolor='none', edgecolor='black', linewidth=3)
    ccds_col = PatchCollection(ccds, facecolor='none', edgecolor='blue', linewidth=2, linestyle='dashed')
    amps_col = PatchCollection(amps, facecolor='none', edgecolor='red', linewidth=1, linestyle='dotted')

    return amps_col, ccds_col, rafts_col


def show_camera(axis, camera):
    """ Add camera to an axis

    Parameters
    ----------
    axis : `matplotlib.axes.Axes`
        the matplotlib figure axis on which to add the camera
    camera : `tuple`
        a tuple of collections containing the camera rectangles

    Returns
    -------
    """
    for col in camera:
        axis.add_collection(col)
    return axis
