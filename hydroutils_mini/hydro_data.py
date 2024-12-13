import numpy as np
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