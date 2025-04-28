"""
@Author:             Yikai CHAI
@Email:              chaiyikai@mail.dlut.edu.cn
@Company:            Dalian University of Technology
@Date:               2025-03-23 19:03:21
@Last Modified by:   Yikai CHAI
@Last Modified time: 2025-04-26 16:24:15
"""

import datetime
from typing import Union
import numpy as np
import pandas as pd


def t2str(t_: Union[str, datetime.datetime]):
    if type(t_) is str:
        return datetime.datetime.strptime(t_, "%Y-%m-%d")
    elif type(t_) is datetime.datetime:
        return t_.strftime("%Y-%m-%d")
    else:
        raise NotImplementedError("We don't support this data type yet")


def t_range_days(t_range, *, step=np.timedelta64(1, "D")) -> np.array:
    """
    Transform the two-value t_range list to a uniformly-spaced list (default is a daily list).
    For example, ["2000-01-01", "2000-01-05"] -> ["2000-01-01", "2000-01-02", "2000-01-03", "2000-01-04"]
    Parameters
    ----------
    t_range
        two-value t_range list
    step
        the time interval; its default value is 1 day
    Returns
    -------
    np.array
        a uniformly-spaced (daily) list
    """
    startdata = datetime.datetime.strptime(t_range[0], "%Y-%m-%d")
    enddata = datetime.datetime.strptime(t_range[1], "%Y-%m-%d")
    return np.arange(startdata, enddata, step)


def t_range_hours(t_range, *, step=np.timedelta64(1, "h")) -> np.array:
    """
    Transform the two-value t_range list to a uniformly-spaced list with hourly intervals.
    For example, ["2000-01-01 00:00", "2000-01-01 05:00"] -> 
    ["2000-01-01 00:00", "2000-01-01 01:00", "2000-01-01 02:00", "2000-01-01 03:00", "2000-01-01 04:00"]
    
    Parameters
    ----------
    t_range
        two-value t_range list with time format "YYYY-MM-DD HH:MM"
    step
        the time interval; its default value is 1 hour
        
    Returns
    -------
    np.array
        a uniformly-spaced (hourly) list
    """
    startdata = datetime.datetime.strptime(t_range[0], "%Y-%m-%d %H:%M")
    enddata = datetime.datetime.strptime(t_range[1], "%Y-%m-%d %H:%M")
    return np.arange(startdata, enddata, step)


def read_merge_timestep(nt, data_temp, t_lst, obs):
    result = np.full([nt], np.nan)
    df_date = data_temp[[0, 1, 2]]
    df_date.columns = ["year", "month", "day"]
    date = pd.to_datetime(df_date).values.astype("datetime64[D]")
    _, ind1, ind2 = np.intersect1d(date, t_lst, return_indices=True)
    result[ind2] = obs[ind1]
    return result
