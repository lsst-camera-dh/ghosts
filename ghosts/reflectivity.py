"""reflectivity module

This module is used to read and handcraft wave length dependent reflectivity of the different
optical element

"""


import batoid
import pandas as pd


def read_sensor_reflectivity(file_path='../data/sensor_reflectivity_qe_r+qe.xlsx'):
    """ Reads sensor reflectivity from a spreadsheet and get back data formatted into a panda data frame

    Parameters
    ----------
    file_path : `string`
        the path to the spreadsheet containing sensor reflectivity data

    Returns
    -------
    data_frame : `pandas.DataFrame`
        a panda data frame with wavelength, reflectivity, Q.E. and Q.E. error
    """
    # read spreadsheet
    df = pd.read_excel(file_path)
    # fix headers
    keys = df.keys()
    df_r = df.rename(columns={keys[0]: "wavelength", keys[1]: "reflectivity",
                              keys[2]: "QE", keys[3]: "sum_qe_r"})
    # clean up data frame
    data_frame = df_r.iloc[1:, :]
    return data_frame


def make_simple_coating(telescope, r_frac=[0.02, 0.02, 0.15], debug=False):
    """ Applies a simple coating as a unique refraction index for each optical element surface

    Parameters
    ----------
    telescope : `batoid.telescope`
        the optical setup as defined in `batoid`
    r_frac : `list` of `float`
        a refraction index, usually of the order of 0.02
    debug : `bool`
        print debug information or not

    Returns
    -------
    coef : `float`
        the reflexion coefficient applied
    """
    coef = r_frac[0]
    for surface in telescope.itemDict.values():
        if isinstance(surface, batoid.RefractiveInterface):
            surface.forwardCoating = batoid.SimpleCoating(coef, 1 - coef)
            surface.reverseCoating = batoid.SimpleCoating(coef, 1 - coef)
        if isinstance(surface, batoid.Detector):
            surface.forwardCoating = batoid.SimpleCoating(coef, 1 - coef)
    return coef


def make_smart_coating(telescope, r_frac=[0.02, 0.02, 0.15], debug=False):
    """ Applies a different reflexion index for each element type lens, filter, detector

    Parameters
    ----------
    telescope : `batoid.telescope`
        the optical setup as defined in `batoid`
    r_frac : `list` of `float`
        the fraction of light that you wish surfaces to reflect, usually of the order of 0.02
        use a list of a unique element for simple coating, or of 3 elements for smart coating (lens, filter, detector)
    debug : `bool`
        print debug information or not

    Returns
    -------
    """
    r_lens, r_filter, r_detector = r_frac[0], r_frac[1], r_frac[2]
    if debug:
        print("Smart coating: ", r_lens, r_filter, r_detector)
    for surface in telescope.itemDict.values():
        if isinstance(surface, batoid.RefractiveInterface):
            if surface.name.split('_')[0] in ['L1', 'L2', 'L3']:
                surface.forwardCoating = batoid.SimpleCoating(r_lens, 1 - r_lens)
                surface.reverseCoating = batoid.SimpleCoating(r_lens, 1 - r_lens)
            elif surface.name.split('_')[0] in ['Filter']:
                surface.forwardCoating = batoid.SimpleCoating(r_filter, 1 - r_filter)
                surface.reverseCoating = batoid.SimpleCoating(r_filter, 1 - r_filter)
        if isinstance(surface, batoid.Detector):
            surface.forwardCoating = batoid.SimpleCoating(r_detector, 1 - r_detector)
    return 0
