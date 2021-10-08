import numpy as np


# define function to get nice rangs
def get_ranges(x, y, dr=0.010):
    x_min = x.mean() - dr
    x_max = x.mean() + dr
    y_min = y.mean() - dr
    y_max = y.mean() + dr
    return (x_min, x_max, y_min, y_max)

def get_main_impact_point(rForward):
    ''' Return main image light rays

    Direct path will be rForward with fewest number of things in "path"
    '''
    i_straight = np.argmin([len(rrr.path) for rrr in rForward])
    direct_x = np.mean(rForward[i_straight].x)
    direct_y = np.mean(rForward[i_straight].y)
    direct_f = rForward[i_straight].flux[0]  # these are all equal
    return i_straight, direct_x, direct_y, direct_f

