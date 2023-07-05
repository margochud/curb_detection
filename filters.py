import numpy as np
import tqdm


def avg_left(coord, k=5):
    r"""Calculates Avg(Left) according to CurbScan article

    Avg(left) stands for the average left directions from point
    cloud dots. For the details go to Eq.2 in the article.

    Parameters
    ----------
    coord: array_like
    Coordinates of the point cloud dots
    k: int
    Number of dots to use for calculation

    Returns
    -------
    avg: np.array
    Avg(Left) for each dot in point cloud
    """
    n = coord.shape
    avg = np.zeros(n)
    print('Calculation of Avg(left):')
    for i in tqdm.trange(1, n[0]):
        coord_left = coord[max(i - k, 0):i]
        avg[i] = np.sum(coord_left - coord[i], axis=0)
    return avg


def avg_right(coord, k=5):
    r"""Calculates Avg(Right) according to CurbScan article

    Avg(Right) stands for the average right directions from point
    cloud dots. For the details go to Eq.2 in the article.

    Parameters
    ----------
    coord: array_like
    Coordinates of the point cloud dots
    k: int
    Number of dots to use for calculation

    Returns
    -------
    avg: np.array
    Avg(Right) for each dot in point cloud
    """
    n = coord.shape
    avg = np.zeros(n)
    print('Calculation of Avg(Right):')
    for i in tqdm.trange(1, n[0]):
        coord_right = coord[i + 1:i + k + 1]
        avg[i] = np.sum(coord_right - coord[i], axis=0)
    return avg


def calc_theta(avg_l, avg_r):
    r"""Calculates angle Theta according to CurbScan article

    Theta stands for vertical angle of the laser beam.
    For the details go to Eq.1 in the article.

    Parameters
    ----------
    avg_l: array_like
    Previously calculated Avg(Left)
    avg_r: array_like
    Previously calculated Avg(Right)

    Returns
    -------
    theta: np.array
    Theta for each dot in point cloud
    """
    numer = np.sum(avg_l * avg_r, axis=1)
    denom = np.linalg.norm(avg_l, axis=1) * np.linalg.norm(avg_r, axis=1)
    theta = np.arccos(numer / (denom + 1e-10))
    return theta


def is_local_min(theta):
    r"""Indicates if angle Theta is local minimum

    Parameters
    ----------
    theta: Theta angle

    Returns
    -------
    np.array, boolean
    Boolean array, where i-th element tells if theta[i]
    is local minimum.
    """
    res = [False for _ in range(len(theta))]
    for i in range(1, len(theta) - 1):
        if theta[i] < theta[i - 1] and theta[i] < theta[i + 1]:
            res[i] = True
    return np.array(res)


def dir_change_filter(coord, thresh=2.0, k=10):
    r"""Direction change filter from CurbScan.

    Detects the elevation deviations where curbs are present.
    For the details go to Algorithm 1 in the article.

    Parameters
    ----------
    coord: array_like
    Coordinates of the point cloud dots.
    thresh: float
    Filter threshold, which is chosen empirically.
    k:int
    Number of dots to use for calculation of
    the average directions from point cloud dots.

    Returns
    -------
    np.array, boolean
    Filtering result.
    """
    print('Direction change filter in progress...')
    avg_r = avg_right(coord, k)
    avg_l = avg_left(coord, k)
    theta = calc_theta(avg_l, avg_r)
    theta_min = is_local_min(theta)
    return (theta < thresh) & theta_min


def elevation_filter(coord, thresh=130):
    r"""Elevation filter from CurbScan.

    Detects the elevation change of consecutive points
    in each scan line.
    For the details go to Algorithm 2 in the article.

    Parameters
    ----------
    coord: array_like
    Coordinates of the point cloud dots.
    thresh: float
    Filter threshold, which is chosen empirically.

    Returns
    -------
    np.array, boolean
    Filtering result.
    """
    res = [False]
    print('Elevation filter in progress...')
    for i in tqdm.trange(1, len(coord)):
        if coord[i, 2] - coord[i - 1, 2] > thresh:
            res.append(True)
        else:
            res.append(False)
    return np.array(res)


def cont_filter(coord, version, angles=None, thresh=None, h_s=1.5):
    r"""Continuous filter from CurbScan or
    the Road-Segmentation-Based article.

    Calculates the distance between two consecutive points
    along each scan line. If the distance is higher than
    a threshold, both points are marked as discontinuous points.

    For the details go to Algorithm 4 in the article CurbScan or
    to Step 1 in CurbDetection in the Road-Segmentation-Based article.

    Parameters
    ----------
    coord: array_like
    Coordinates of the point cloud dots.
    version: ['CurbScan', 'RoadSegm'], optional
    Indicates which article implementation to use
    thresh: float
    Filter threshold, which is chosen empirically.
    h_s: float
    The sensor height, valid only for RoadSegm.

    Returns
    -------
    np.array, boolean
    Filtering result.
    """
    if version == 'CurbScan':
        delta = [thresh for _ in range(coord.shape[0])]
    else:
        delta = h_s * np.pi * 0.4 / ((np.tan(angles) + 1e-10) * 180)
    res = [False]
    print('Continuous filter in progress...')
    for i in tqdm.trange(1, len(coord)):
        if np.linalg.norm(coord[i, :2] - coord[i - 1, :2]) < delta[i]:
            res.append(True)
        else:
            res.append(False)
    return np.array(res)
