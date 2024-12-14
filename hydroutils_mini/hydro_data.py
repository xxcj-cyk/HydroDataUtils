import numpy as np
import xarray as xr
from typing import OrderedDict
import warnings


def warn_if_nan(dataarray, max_display=5, nan_mode="any"):
    """
    Issue a warning if the dataarray contains any NaN values and display their locations.

    Parameters
    -----------
    dataarray: xr.DataArray
        Input dataarray to check for NaN values.
    max_display: int
        Maximum number of NaN locations to display in the warning.
    nan_mode: str
        Mode of NaN checking: 'any' for any NaNs, 'all' for all values being NaNs.
    """
    if dataarray is None:
        return
    if nan_mode not in ["any", "all"]:
        raise ValueError("nan_mode must be 'any' or 'all'")

    if nan_mode == "all" and np.all(np.isnan(dataarray.values)):
        raise ValueError("The dataarray contains only NaN values!")

    nan_indices = np.argwhere(np.isnan(dataarray.values))
    total_nans = len(nan_indices)

    if total_nans <= 0:
        return False
    message = f"The dataarray contains {total_nans} NaN values!"

    # Displaying only the first few NaN locations if there are too many
    display_indices = nan_indices[:max_display].tolist()
    message += (
        f" Here are the indices of the first {max_display} NaNs: {display_indices}..."
        if total_nans > max_display
        else f" Here are the indices of the NaNs: {display_indices}"
    )
    warnings.warn(message)

    return True


def wrap_t_s_dict(data_cfgs: dict, is_tra_val_te: str) -> OrderedDict:
    """
    Basins and periods

    Parameters
    ----------
    data_cfgs
        configs for reading from data source
    is_tra_val_te
        train, valid or test

    Returns
    -------
    OrderedDict
        OrderedDict(sites_id=basins_id, t_final_range=t_range_list)
    """
    basins_id = data_cfgs["object_ids"]
    if type(basins_id) is str and basins_id == "ALL":
        raise ValueError("Please specify the basins_id in configs!")
    if any(x >= y for x, y in zip(basins_id, basins_id[1:])):
        # raise a warning if the basins_id is not sorted
        warnings.warn("The basins_id is not sorted!")
    if f"t_range_{is_tra_val_te}" in data_cfgs:
        t_range_list = data_cfgs[f"t_range_{is_tra_val_te}"]
    else:
        raise KeyError(f"Error! The mode {is_tra_val_te} was not found. Please add it.")
    return OrderedDict(sites_id=basins_id, t_final_range=t_range_list)



def unify_streamflow_unit(ds: xr.Dataset, area=None, inverse=False):
    """Unify the unit of xr_dataset to be mm/day in a basin or inverse

    Parameters
    ----------
    ds: xarray dataset
        _description_
    area:
        area of each basin

    Returns
    -------
    _type_
        _description_
    """
    # use pint to convert unit
    if not inverse:
        target_unit = "mm/d"
        q = ds.pint.quantify()
        a = area.pint.quantify()
        r = q[list(q.keys())[0]] / a[list(a.keys())[0]]
        result = r.pint.to(target_unit).to_dataset(name=list(q.keys())[0])
    else:
        target_unit = "m^3/s"
        r = ds.pint.quantify()
        a = area.pint.quantify()
        q = r[list(r.keys())[0]] * a[list(a.keys())[0]]
        # q = q.pint.quantify()
        result = q.pint.to(target_unit).to_dataset(name=list(r.keys())[0])
    # dequantify to get normal xr_dataset
    return result.pint.dequantify()


def wrap_t_s_dict(data_cfgs: dict, is_tra_val_te: str) -> OrderedDict:
    """
    Basins and periods

    Parameters
    ----------
    data_cfgs
        configs for reading from data source
    is_tra_val_te
        train, valid or test

    Returns
    -------
    OrderedDict
        OrderedDict(sites_id=basins_id, t_final_range=t_range_list)
    """
    basins_id = data_cfgs["object_ids"]
    if type(basins_id) is str and basins_id == "ALL":
        raise ValueError("Please specify the basins_id in configs!")
    if any(x >= y for x, y in zip(basins_id, basins_id[1:])):
        # raise a warning if the basins_id is not sorted
        warnings.warn("The basins_id is not sorted!")
    if f"t_range_{is_tra_val_te}" in data_cfgs:
        t_range_list = data_cfgs[f"t_range_{is_tra_val_te}"]
    else:
        raise KeyError(f"Error! The mode {is_tra_val_te} was not found. Please add it.")
    return OrderedDict(sites_id=basins_id, t_final_range=t_range_list)


def _trans_norm(
    x: xr.DataArray,
    var_lst: list,
    stat_dict: dict,
    log_norm_cols: list = None,
    to_norm: bool = True,
    **kwargs,
) -> np.array:
    """
    Normalization or inverse normalization

    There are two normalization formulas:

    .. math:: normalized_x = (x - mean) / std

    and

     .. math:: normalized_x = [log_{10}(\sqrt{x} + 0.1) - mean] / std

     The later is only for vars in log_norm_cols; mean is mean value; std means standard deviation

    Parameters
    ----------
    x
        data to be normalized or denormalized
    var_lst
        the type of variables
    stat_dict
        statistics of all variables
    log_norm_cols
        which cols use the second norm method
    to_norm
        if true, normalize; else denormalize

    Returns
    -------
    np.array
        normalized or denormalized data
    """
    if x is None:
        return None
    if log_norm_cols is None:
        log_norm_cols = []
    if type(var_lst) is str:
        var_lst = [var_lst]
    out = xr.full_like(x, np.nan)
    for item in var_lst:
        stat = stat_dict[item]
        if to_norm:
            out.loc[dict(variable=item)] = (
                (np.log10(np.sqrt(np.abs(x.sel(variable=item))) + 0.1) - stat[2])
                / stat[3]
                if item in log_norm_cols
                else (x.sel(variable=item) - stat[2]) / stat[3]
            )
        elif item in log_norm_cols:
            out.loc[dict(variable=item)] = (
                np.power(10, x.sel(variable=item) * stat[3] + stat[2]) - 0.1
            ) ** 2
        else:
            out.loc[dict(variable=item)] = x.sel(variable=item) * stat[3] + stat[2]
    if to_norm:
        # after normalization, all units are dimensionless
        out.attrs = {}
    # after denormalization, recover units
    else:
        if "recover_units" in kwargs.keys() and kwargs["recover_units"] is not None:
            recover_units = kwargs["recover_units"]
            for item in var_lst:
                out.attrs["units"][item] = recover_units[item]
    return out


def _prcp_norm(x: np.array, mean_prep: np.array, to_norm: bool) -> np.array:
    """
    Normalize or denormalize data with mean precipitation.

    The formula is as follows when normalizing (denormalize equation is its inversion):

    .. math:: normalized_x = \frac{x}{precipitation}

    Parameters
    ----------
    x
        data to be normalized or denormalized
    mean_prep
        basins' mean precipitation
    to_norm
        if true, normalize; else denormalize

    Returns
    -------
    np.array
        normalized or denormalized data
    """
    tempprep = np.tile(mean_prep, (1, x.shape[1]))
    return x / tempprep if to_norm else x * tempprep