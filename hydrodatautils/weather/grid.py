"""
此代码用于网格化数据, 包括将数据转换为nc文件, 将数据插值后转换为nc文件, 生成网格, 生成mask等功能
编写：柴熠垲
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
    将dataframe数据转换为nc文件

    Args:
        filename (str): 目标 .nc 文件的文件名
        src (pd.DataFrame): 包含 x, y 轴坐标和变量值的源 DataFrame
        x_axis (str): DataFrame 中作为 x 轴的列名，默认为 'x'
        y_axis (str): DataFrame 中作为 y 轴的列名，默认为 'y'
        variable (str): DataFrame 中作为变量值的列名，默认为 'value'

    Returns:
        xarray.Dataset: 读取 .nc 文件后的 xarray 数据集对象
    """

    data = nc.Dataset(filename, "w", format="NETCDF4")

    x_dim = np.sort(np.unique(src[x_axis]))
    y_dim = np.sort(np.unique(src[y_axis]))[::-1]

    # 创建维度，第一个参数为维度名，第二个参数为维度长度
    data.createDimension(x_axis, x_dim.shape[0])
    data.createDimension(y_axis, y_dim.shape[0])

    # 创建变量，变量部分不需要传输数据
    x = data.createVariable(x_axis, np.float64, (x_axis,))
    y = data.createVariable(y_axis, np.float64, (y_axis,))
    value = data.createVariable(variable, np.float32, (y_axis, x_axis))

    # 把有数据的nc文件，赋值给创建的nc文件
    x[:] = x_dim[:]
    y[:] = y_dim[:]

    array = np.full((y_dim.shape[0], x_dim.shape[0]), -1, dtype=np.float32)
    for index, row in src.iterrows():
        x_index = np.where(x_dim == row[x_axis])[0][0]
        y_index = np.where(y_dim == row[y_axis])[0][0]
        array[y_index, x_index] = row[variable]

    value[:, :] = array[:, :]

    # 最后把data关闭
    data.close()

    return xr.open_dataset(filename, engine="netcdf4")


def xarray_to_lonlat(xds, x_axis="x", y_axis="y"):
    x_dim = xds[x_axis].to_numpy()
    y_dim = xds[y_axis].to_numpy()

    return x_dim, y_dim


def idw_to_xarray(filename, df, array, x_axis="x", y_axis="y", variable="value"):
    """
    将dataframe数据idw插值后转换为nc文件
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
    用于确定边界所在网格的中心点坐标

    Args:
        v (float): 边界点的坐标
        resolution (float): 网格的分辨率
        offset (float): 网格的偏移量
        plusone (bool, 可选): 一个标志位, 用于决定在某些边界条件下是否需要将返回的中心点坐标增加一个网格的分辨率, 默认为 False

    Returns:
        float: 网格中心点坐标
    """
    center = (
        math.floor(v) - offset
    )  # center为v向下取整后减去偏移量offset的值；寻找离v最近的可能的网格中心点
    while center <= math.ceil(v) + offset:  # 找到满足条件的网格中心或者超过v的可能范围
        if (
            abs(center - v) <= resolution / 2
        ):  # 如果center与v的差的绝对值小于或等于网格分辨率的一半，说明v位于当前center代表的网格内
            if (
                abs(abs(center - v) - resolution / 2) < 0.00001 and plusone
            ):  # 如果v正好在网格的边缘，并且plusone为True，则将中心点坐标增加一个网格的分辨率
                return center + resolution
            else:
                return center  # 其他情况下，返回当前的中心点坐标
        center += resolution  # 如果当前的center不满足条件，增加resolution以尝试下一个可能的网格中心
    return None  # 如果函数未能在循环中返回任何中心点坐标，最终返回None，表示没有找到满足条件的网格中心


def gen_grids(bbox, resolution, offset, x_axis="x", y_axis="y"):
    """
    用于生成geopandas格式网格

    Args:
        bbox (tuple): 流域的边界框
        resolution (float): 网格的分辨率
        offset (tuple): 网格的偏移量
        x_axis (str, 可选): 用作X轴的列名, 默认为"x"
        y_axis (str, 可选): 用作Y轴的列名, 默认为"y"

    Returns:
        gpd.GeoDataFrame: 包含网格信息的地理数据框
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
    基于流域数据生成空间覆盖(mask)文件

    Args:
        watershed (GeoDataFrame): 输入的地理数据框, 应包含流域的地理信息和几何形状
        resolution (float): 网格的分辨率, 单位是度
        offset (tuple): 网格的偏移量, 单位是度
        filename (str): 生成的NetCDF文件的文件名
        x_axis (str, 可选): 用作X轴的列名, 默认为"x"
        y_axis (str, 可选): 用作Y轴的列名, 默认为"y"

    Returns:

    """

    for index, row in watershed.iterrows():
        # wid = row[fieldname]
        geo = row["geometry"]  # 获取流域的几何形状
        bbox = geo.bounds  # 获取流域的边界框

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
