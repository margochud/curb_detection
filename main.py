import os
import sys
from datetime import datetime

import laspy
import numpy as np

import filters
from helper_ply import read_ply


def paths(sys_argv):
    r"""Finds the paths for input and output data
        Parameters
        ----------
        sys_argv: array_like
        Contains all the command-line arguments.

        Returns
        -------
        path_in: str
        Path to input data.
        path_out: str
        Path to output data.
        """

    path_in = sys_argv[1]
    if len(sys.argv) == 3:
        path_out = sys_argv[2]
    else:
        home_path = os.path.expanduser('~')
        cur_time = datetime.now()
        time_format = "%Y-%m-%d %H:%M:%S"
        path_out = home_path + r'/Desktop/borders_' + f"{cur_time:{time_format}}.las"
    return path_in, path_out


def transform_to_las(new_data, path):
    r"""Saves data as a .las file
        Parameters
        ----------
        new_data: array_like
        Array with characteristics of extracted data.
        path:str
        Path to the .las file.

        Returns
        -------
        None
        """
    header = laspy.LasHeader(point_format=6, version="1.4")
    header.offsets = np.min(coord, axis=0)
    header.scales = np.array([0.1, 0.1, 0.1])
    las = laspy.LasData(header)

    las.x = new_data['x']
    las.y = new_data['y']
    las.z = new_data['z']
    las.Red = new_data['red']
    las.Green = new_data['green']
    las.Blue = new_data['blue']
    las.Intensity = new_data['scalar_Intensity']
    las.gps_time = new_data['scalar_GPSTime']
    las.scan_angle = new_data['scalar_ScanAngleRank']
    las.classification = new_data['scalar_Label']
    las.write(path)


if __name__ == '__main__':
    # paths to files
    path_in, path_out = paths(sys.argv)

    # read file and get coord
    data = read_ply(path_in)
    road = (data['scalar_Label'] == 1) | (data['scalar_Label'] == 2)
    data = data[road]
    coord = np.vstack((data['x'], data['y'], data['z'])).T

    # filters constants
    k = 25
    thresh_dir = 2.0
    thresh_el = 0.03
    thresh_cont = 0.5

    # apply filters
    properties = data.dtype.names
    res_dir = filters.dir_change_filter(coord, thresh=thresh_dir, k=k)
    res_el = filters.elevation_filter(coord, thresh=thresh_el)
    res_cont = filters.cont_filter(coord, version='CurbScan', thresh=thresh_cont)
    res_avg = res_dir & res_el & res_cont
    new_data = data[res_avg]

    # transform to las
    transform_to_las(new_data, path_out)
