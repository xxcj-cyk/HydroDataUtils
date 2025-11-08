"""
This code is used for gridding data, including converting data to nc files, 
converting data to nc files after interpolation, generating grids, generating masks and other functions.
Author: Yikai CHAI
"""

import math
import numpy as np
import pandas as pd
import netCDF4 as nc
import xarray as xr
import geopandas as gpd
from shapely.geometry import Polygon
from .interpolation import IDW


def to_xarray(filename, src: pd.DataFrame, x_axis="x", y_axis="y", variable="value"):
    """
    Convert dataframe data to nc file

    Args:
        filename (str): The filename of the target .nc file
        src (pd.DataFrame): Source DataFrame containing x, y axis coordinates and variable values
        x_axis (str): Column name in DataFrame used as x-axis, default is 'x'
        y_axis (str): Column name in DataFrame used as y-axis, default is 'y'
        variable (str): Column name in DataFrame used as variable value, default is 'value'

    Returns:
        xarray.Dataset: xarray dataset object after reading the .nc file
    """

    data = nc.Dataset(filename, "w", format="NETCDF4")

    x_dim = np.sort(np.unique(src[x_axis]))
    y_dim = np.sort(np.unique(src[y_axis]))[::-1]

    # Create dimension, first parameter is dimension name, second parameter is dimension length
    data.createDimension(x_axis, x_dim.shape[0])
    data.createDimension(y_axis, y_dim.shape[0])

    # Create variable, variable part doesn't need to transmit data
    x = data.createVariable(x_axis, np.float64, (x_axis,))
    y = data.createVariable(y_axis, np.float64, (y_axis,))
    value = data.createVariable(variable, np.float32, (y_axis, x_axis))

    # Assign the nc file with data to the created nc file
    x[:] = x_dim[:]
    y[:] = y_dim[:]

    array = np.full((y_dim.shape[0], x_dim.shape[0]), -1, dtype=np.float32)
    for index, row in src.iterrows():
        x_index = np.where(x_dim == row[x_axis])[0][0]
        y_index = np.where(y_dim == row[y_axis])[0][0]
        array[y_index, x_index] = row[variable]

    value[:, :] = array[:, :]

    # Finally close the data
    data.close()

    return xr.open_dataset(filename, engine="netcdf4")


def xarray_to_lonlat(xds, x_axis="x", y_axis="y"):
    x_dim = xds[x_axis].to_numpy()
    y_dim = xds[y_axis].to_numpy()

    return x_dim, y_dim


def idw_to_xarray(filename, df, array, x_axis="x", y_axis="y", variable="value"):
    """
    Convert dataframe data to nc file after IDW interpolation
    """
    x = df[x_axis].to_numpy()
    y = df[y_axis].to_numpy()
    z = df[variable].to_numpy()
    grid_lon_list, grid_lat_list = xarray_to_lonlat(array, x_axis=x_axis, y_axis=y_axis)

    pm_idw = IDW(x, y, z, grid_lon_list, grid_lat_list)

    IDW_grid_df = pd.DataFrame(pm_idw, columns=[x_axis, y_axis, variable])
    return to_xarray(
        filename, IDW_grid_df, x_axis=x_axis, y_axis=y_axis, variable=variable
    )


def find_center(v, resolution, offset, plusone=False):
    """
    Determine the center point coordinates of the grid where the boundary is located

    Args:
        v (float): Coordinate of the boundary point
        resolution (float): Resolution of the grid
        offset (float): Offset of the grid
        plusone (bool, optional): A flag bit to determine whether to increase the returned center point coordinate by one grid resolution under certain boundary conditions, default is False

    Returns:
        float: Grid center point coordinate
    """
    center = (
        math.floor(v) - offset
    )  # center is the value of v rounded down minus the offset; find the nearest possible grid center point to v
    while center <= math.ceil(v) + offset:  # Find the grid center that satisfies the condition or exceeds the possible range of v
        if (
            abs(center - v) <= resolution / 2
        ):  # If the absolute value of the difference between center and v is less than or equal to half of the grid resolution, v is located within the grid represented by the current center
            if (
                abs(abs(center - v) - resolution / 2) < 0.00001 and plusone
            ):  # If v is exactly at the edge of the grid and plusone is True, increase the center point coordinate by one grid resolution
                return center + resolution
            else:
                return center  # In other cases, return the current center point coordinate
        center += resolution  # If the current center doesn't satisfy the condition, increase resolution to try the next possible grid center
    return None  # If the function fails to return any center point coordinate in the loop, return None, indicating that no grid center satisfying the condition was found


def gen_grids(bbox, resolution, offset, x_axis="x", y_axis="y"):
    """
    Generate grids in geopandas format

    Args:
        bbox (tuple): Boundary box of the watershed
        resolution (float): Resolution of the grid
        offset (tuple): Offset of the grid
        x_axis (str, optional): Column name used as X-axis, default is "x"
        y_axis (str, optional): Column name used as Y-axis, default is "y"

    Returns:
        gpd.GeoDataFrame: GeoDataFrame containing grid information
    """

    lx = bbox[0]
    rx = bbox[2]
    LLON = find_center(lx, resolution, offset, plusone=True)
    RLON = find_center(rx, resolution, offset)
    by = bbox[1]
    ty = bbox[3]
    BLAT = find_center(by, resolution, offset, plusone=True)
    TLAT = find_center(ty, resolution, offset)

    # print(LLON,BLAT,RLON,TLAT)

    xsize = round((RLON - LLON) / resolution) + 1
    ysize = round((TLAT - BLAT) / resolution) + 1

    # print(xsize, ysize)

    lons = np.linspace(LLON, RLON, xsize)
    lats = np.linspace(TLAT, BLAT, ysize)

    geometry = []
    HBlons = []
    HBlats = []

    for i in range(xsize):
        for j in range(ysize):
            HBLON = lons[i]
            HBLAT = lats[j]

            HBlons.append(HBLON)
            HBlats.append(HBLAT)

            geometry.append(
                Polygon(
                    [
                        ((HBLON - resolution / 2), (HBLAT + resolution / 2)),
                        ((HBLON + resolution / 2), (HBLAT + resolution / 2)),
                        ((HBLON + resolution / 2), (HBLAT - resolution / 2)),
                        ((HBLON - resolution / 2), (HBLAT - resolution / 2)),
                    ]
                )
            )

    data = gpd.GeoDataFrame(crs="EPSG:4326", geometry=geometry)
    data[x_axis] = HBlons
    data[y_axis] = HBlats

    return data


def gen_mask(watershed, resolution, offset, filename, x_axis="x", y_axis="y"):
    """
    Generate spatial coverage (mask) file based on watershed data

    Args:
        watershed (GeoDataFrame): Input GeoDataFrame, should contain geographic information and geometry of the watershed
        resolution (float): Resolution of the grid, unit is degrees
        offset (tuple): Offset of the grid, unit is degrees
        filename (str): Filename of the generated NetCDF file
        x_axis (str, optional): Column name used as X-axis, default is "x"
        y_axis (str, optional): Column name used as Y-axis, default is "y"

    Returns:

    """

    for index, row in watershed.iterrows():
        # wid = row[fieldname]
        geo = row["geometry"]  # Get the geometry of the watershed
        bbox = geo.bounds  # Get the boundary box of the watershed

        grid = gen_grids(bbox, resolution, offset, x_axis=x_axis, y_axis=y_axis)
        grid = grid.to_crs(epsg=3857)
        grid["GRID_AREA"] = grid.area
        grid = grid.to_crs(epsg=4326)

        gs = gpd.GeoSeries.from_wkt([geo.wkt])
        sub = gpd.GeoDataFrame(crs="EPSG:4326", geometry=gs)

        intersects = gpd.overlay(grid, sub, how="intersection")
        intersects = intersects.to_crs(epsg=3857)
        intersects["BASIN_AREA"] = intersects.area
        intersects = intersects.to_crs(epsg=4326)
        intersects["w"] = intersects["BASIN_AREA"] / intersects["GRID_AREA"]

        grids = grid.set_index([x_axis, y_axis]).join(
            intersects.set_index([x_axis, y_axis]), lsuffix="_left", rsuffix="_right"
        )
        grids = grids.loc[:, ["w"]]
        grids.loc[grids.w.isnull(), "w"] = 0

        wds = grids.to_xarray()
        wds.to_netcdf(f"{filename}{index}")


if __name__ == "__main__":
    pass
